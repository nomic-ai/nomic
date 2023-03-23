import json
import os
import time
from pathlib import Path

import click
import requests
from rich.console import Console

tenants = {
    'staging': {'frontend_domain': 'staging-atlas.nomic.ai', 'api_domain': 'staging-api-atlas.nomic.ai'},
    'production': {'frontend_domain': 'atlas.nomic.ai', 'api_domain': 'api-atlas.nomic.ai'},
}

nomic_base_path = Path.home() / '.nomic'



def validate_api_http_response(response):
    if response.status_code >= 500 and response.status_code < 600:
        raise Exception("Cannot contact establish a connection with Nomic services.")

    return response


def get_api_credentials(fn=None):
    if fn is None:
        fn = 'credentials'
    filepath = nomic_base_path / fn
    if not filepath.exists():
        raise ValueError("You have not configured your Nomic API token. Run `nomic login` to configure.")

    with open(filepath, 'r') as file:
        credentials = json.load(file)
        return credentials


def login(token, tenant='production'):
    environment = tenants[tenant]
    auth0_auth_endpoint = f"https://{environment['frontend_domain']}/cli-login"

    console = Console()
    style = "bold"
    if not token:
        console.print("Authenticate with the Nomic API", style=style, justify="center")
        console.print(auth0_auth_endpoint, style=style, justify="center")
        console.print(
            "Click the above link to retrieve your access token and then run `nomic login \[token]`",
            style=style,
            justify="center",
        )
        exit()

    # save credential
    if not nomic_base_path.exists():
        nomic_base_path.mkdir()

    response = requests.get('https://' + environment['api_domain'] + f"/v1/user/token/refresh/{token}")
    response = validate_api_http_response(response)

    if not response.status_code == 200:
        raise Exception("Could not authorize you with Nomic. Run `nomic login` to re-authenticate.")

    bearer_token = response.json()['access_token']
    with open(os.path.join(nomic_base_path, 'credentials'), 'w') as file:
        json.dump(
            {'refresh_token': token, 'token': bearer_token, 'tenant': tenant, 'expires': time.time() + 80000}, file
        )


def refresh_bearer_token():
    credentials = get_api_credentials()
    if time.time() >= credentials['expires']:
        environment = tenants[credentials['tenant']]
        response = requests.get(
            'https://' + environment['api_domain'] + f"/v1/user/token/refresh/{credentials['refresh_token']}"
        )
        response = validate_api_http_response(response)

        if not response.status_code == 200:
            raise Exception("Could not authorize you with Nomic. Run `nomic login` to re-authenticate.")

        bearer_token = response.json()['access_token']
        credentials['token'] = bearer_token
        with open(os.path.join(nomic_base_path, 'credentials'), 'w') as file:
            json.dump(credentials, file)
    return credentials

def switch(tenant):
    assert tenant in ['staging', 'production', None]
    credentials = get_api_credentials()
    current_tenant = credentials['tenant']
    if tenant is None:
        print(f'Current tenant: {current_tenant}')
        return
    if current_tenant == tenant:
        return
    else:
        current_loc = nomic_base_path / 'credentials'
        new_loc = nomic_base_path / f'credentials_{current_tenant}'
        print(f'Switching from {current_tenant} to {tenant}.')
        if current_loc.exists():
            current_loc.rename(new_loc)
        if (nomic_base_path / f'credentials_{tenant}').exists():
            (nomic_base_path / f'credentials_{tenant}').rename(current_loc)
        else:
            login(token=None, tenant=tenant)
        

@click.command()
@click.argument('command', nargs=1, default='')
@click.argument('params', nargs=-1)
def cli(command, params):
    if command == 'login':
        if len(params) == 0:
            login(token=None, tenant='production')
        if len(params) == 1 and params[0] == 'staging':
            login(token=None, tenant='staging')
        if len(params) == 2 and params[0] == 'staging':
            login(token=params[1], tenant='staging')
        if len(params) == 1:
            login(token=params[0], tenant='production')
    elif command == 'switch':
        if len(params) == 0:
            switch(tenant=None)
        if len(params) == 1:
            switch(tenant=params[0])
    else:
        raise ValueError(f"Command {command} not found.")
    
if __name__ == "__main__":
    cli()
