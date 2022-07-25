import click
from rich.console import Console

auth0_domain: str = 'nomicai.us.auth0.com'
auth0_api_audience: str = 'AtlasAPI'
auth0_client_id: str = 'Gu47wjsnpW2PPfinIHpVnjpVclAnC8k4'
frontend_domain: str = 'atlas.nomic.ai'

auth0_auth_endpoint = f"https://{auth0_domain}/authorize?response_type=token&client_id={auth0_client_id}&redirect_uri=https://{frontend_domain}/token"

def login(token):
    console = Console(width=143)
    style = "bold white on blue"
    if not token:
        console.print("Authorize with the Nomic API", style=style, justify="center")
        console.print(auth0_auth_endpoint, style=style)
        exit()

    #save credential
    

@click.command()
@click.argument('command', nargs=1, default='')
@click.argument('params', nargs=-1)
def cli(command, params):
    if command == 'login':
        login(token=params)


if __name__ == "__main__":
    cli()