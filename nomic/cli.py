import json
import os
from pathlib import Path

import click
from rich.console import Console

tenants = {
    'staging': {
        'auth0_domain': 'nomicai.us.auth0.com',
        'auth0_api_audience': 'AtlasAPI',
        'auth0_client_id': 'Gu47wjsnpW2PPfinIHpVnjpVclAnC8k4',
        'frontend_domain': 'staging-atlas.nomic.ai',
    },
    'production': {
        'auth0_domain': 'nomicai-production.us.auth0.com',
        'auth0_api_audience': 'https://api-atlas.nomic.ai',
        'auth0_client_id': 'VF41DqdEyS2Aaq64q1IoO1EOzdpjplnv',
        'frontend_domain': 'atlas.nomic.ai',
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
    # auth0_auth_endpoint = f"https://{environment['auth0_domain']}/authorize?response_type=code&client_id={environment['auth0_client_id']}&redirect_uri=https://{environment['frontend_domain']}/token&scope=openid+profile+email&audience={environment['auth0_api_audience']}"
    auth0_auth_endpoint = f"https://{environment['frontend_domain']}/cli-login"

    console = Console()
    style = "bold"
    if not token:
        console.print("Authorize with the Nomic API", style=style, justify="center")
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

    with open(os.path.join(nomic_base_path, 'credentials'), 'w') as file:
        json.dump({'token': token, 'tenant': tenant}, file)


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
