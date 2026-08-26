"""
lpix — CLI entry point
"""
import click
from lpix.sync.ingest import main as ingest_main


@click.group()
def main():
    """lpix — Launchpad Bug RAG System"""
    pass


main.add_command(ingest_main, name="ingest")


@main.command()
@click.option("--project", required=True)
def status(project):
    """Show sync status for a project."""
    from lpix.sync.state import SyncState
    from lpix.retrieval.store import BugVectorStore
    
    state = SyncState()
    store = BugVectorStore()
    
    last_sync = state.get_last_sync(project)
    count = state.get_bug_count(project)
    total_vectors = store.count()
    
    click.echo(f"Project: {project}")
    click.echo(f"Last sync: {last_sync or 'never'}")
    click.echo(f"Bug count: {count}")
    click.echo(f"Total vectors in store: {total_vectors}")


@main.command()
@click.argument("query")
@click.option("--n", default=5, type=int)
@click.option("--status", "status_filter", default=None)
@click.option("--importance", "importance_filter", default=None)
@click.option("--project", "project_filter", default=None)
def search(query, n, status_filter, importance_filter, project_filter):
    """Search bugs from the command line."""
    from lpix.tools.search_tool import search_launchpad_bugs
    result = search_launchpad_bugs(
        query=query,
        n_results=n,
        status_filter=status_filter,
        importance_filter=importance_filter,
        project_filter=project_filter,
    )
    click.echo(result)


if __name__ == "__main__":
    main()
