import json
import os
from pathlib import Path

import click
import time
import requests
from rich.console import Console

tenants = {
    'staging': {
        'frontend_domain': 'staging-atlas.nomic.ai',
        'api_domain': 'staging-api-atlas.nomic.ai'
    },
    'production': {
        'frontend_domain': 'atlas.nomic.ai',
        'api_domain': 'api-atlas.nomic.ai'
    },
}

nomic_base_path = f'{str(Path.home())}/.nomic'


def get_api_credentials():
    if not os.path.exists(os.path.join(nomic_base_path, 'credentials')):
        raise ValueError("You have not configured your Nomic API token. Run `nomic login` to configure.")

    with open(os.path.join(nomic_base_path, 'credentials'), 'r') as file:
        credentials = json.load(file)
        return credentials


def login(token, tenant):
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
    if not os.path.exists(nomic_base_path):
        os.mkdir(nomic_base_path)

    response = requests.get(
        'https://'+environment['api_domain'] + f"/v1/user/token/refresh/{token}"
    )
    print('https://'+environment['api_domain'] + f"/v1/user/token/refresh/{token}")

    if not response.status_code == 200:
        print(response.json())
        raise Exception("Could not authorize you with Nomic. Run `nomic login` to re-authenticate.")

    bearer_token = response.json()['access_token']
    with open(os.path.join(nomic_base_path, 'credentials'), 'w') as file:
        json.dump({'refresh_token': token, 'token': bearer_token, 'tenant': tenant, 'expires': time.time()+80000}, file)

def refresh_bearer_token():
    credentials = get_api_credentials()
    if time.time() >= credentials['expires']:
        environment = tenants[credentials['tenant']]
        response = requests.get(
            'https://'+environment['api_domain'] + f"/v1/user/token/refresh/{credentials['refresh_token']}"
        )

        if not response.status_code == 200:
            print(response.json())
            raise Exception("Could not authorize you with Nomic. Run `nomic login` to re-authenticate.")

        bearer_token = response.json()['access_token']
        credentials['access_token'] = bearer_token
        with open(os.path.join(nomic_base_path, 'credentials'), 'w') as file:
            json.dump(credentials, file)


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


if __name__ == "__main__":
    cli()
