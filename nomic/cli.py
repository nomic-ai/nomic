import os
from pathlib import Path

import click
from rich.console import Console

auth0_domain: str = 'nomicai.us.auth0.com'
auth0_api_audience: str = 'AtlasAPI'
auth0_client_id: str = 'Gu47wjsnpW2PPfinIHpVnjpVclAnC8k4'
frontend_domain: str = 'atlas.nomic.ai'

auth0_auth_endpoint = f"https://{auth0_domain}/authorize?response_type=code&client_id={auth0_client_id}&redirect_uri=https://{frontend_domain}/token&scope=openid+profile+email&audience={auth0_api_audience}"

nomic_base_path = f'{str(Path.home())}/.nomic'


def get_api_token():
    if not os.path.exists(os.path.join(nomic_base_path, 'credentials')):
        raise ValueError("You have not configured your Nomic API token. Run `nomic login` to configure.")

    with open(os.path.join(nomic_base_path, 'credentials'), 'r') as file:
        token = str(file.read())
        return token


def login(token):
    console = Console(width=190)
    style = "bold white on blue"
    if not token:
        console.print("Authorize with the Nomic API", style=style, justify="center")
        console.print(auth0_auth_endpoint, style=style)
        console.print(
            "Click the above link to retrieve your access token and then run `nomic login \[token]`",
            style=style,
            justify="center",
        )
        exit()
    token = token[0]
    # save credential
    if not os.path.exists(nomic_base_path):
        os.mkdir(nomic_base_path)

    with open(os.path.join(nomic_base_path, 'credentials'), 'w') as file:
        file.write(token)


@click.command()
@click.argument('command', nargs=1, default='')
@click.argument('params', nargs=-1)
def cli(command, params):
    if command == 'login':
        login(token=params)


if __name__ == "__main__":
    cli()
