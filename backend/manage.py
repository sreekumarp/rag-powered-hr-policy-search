#!/usr/bin/env python
"""
Management CLI for HR Assistant RAG application.

Usage:
    python manage.py create-admin
    python manage.py create-user username email password --role user
    python manage.py list-users
    python manage.py reset-db
"""

import sys
import click
from app import create_app
from models import db, User, Document, QueryLog, APIKey
from services.auth_service import AuthService


app = create_app()


@click.group()
def cli():
    """HR Assistant RAG Management CLI."""
    pass


@cli.command()
@click.option("--username", prompt=True, help="Admin username")
@click.option("--email", prompt=True, help="Admin email")
@click.password_option(help="Admin password")
def create_admin(username, email, password):
    """Create an admin user."""
    with app.app_context():
        success, result = AuthService.register_user(username, email, password, role="admin")
        if success:
            click.echo(click.style(f"✓ Admin user created: {username}", fg="green"))
        else:
            click.echo(click.style(f"✗ Failed: {result.get('error')}", fg="red"))


@cli.command()
@click.argument("username")
@click.argument("email")
@click.argument("password")
@click.option("--role", default="user", type=click.Choice(["user", "admin", "readonly"]))
def create_user(username, email, password, role):
    """Create a new user."""
    with app.app_context():
        success, result = AuthService.register_user(username, email, password, role=role)
        if success:
            click.echo(click.style(f"✓ User created: {username} ({role})", fg="green"))
        else:
            click.echo(click.style(f"✗ Failed: {result.get('error')}", fg="red"))


@cli.command()
def list_users():
    """List all users."""
    with app.app_context():
        users = User.query.all()
        if not users:
            click.echo("No users found.")
            return

        click.echo("\nUsers:")
        click.echo("-" * 80)
        for user in users:
            status = "✓" if user.is_active else "✗"
            click.echo(
                f"{status} {user.username:20} {user.email:30} {user.role:10} "
                f"(ID: {user.id})"
            )
        click.echo("-" * 80)
        click.echo(f"Total: {len(users)} users\n")


@cli.command()
@click.argument("user_id", type=int)
@click.argument("new_role", type=click.Choice(["user", "admin", "readonly"]))
def change_role(user_id, new_role):
    """Change user role."""
    with app.app_context():
        user = User.query.get(user_id)
        if not user:
            click.echo(click.style(f"✗ User not found: {user_id}", fg="red"))
            return

        user.role = new_role
        db.session.commit()
        click.echo(
            click.style(f"✓ Changed {user.username}'s role to {new_role}", fg="green")
        )


@cli.command()
@click.argument("user_id", type=int)
def deactivate(user_id):
    """Deactivate a user."""
    with app.app_context():
        user = User.query.get(user_id)
        if not user:
            click.echo(click.style(f"✗ User not found: {user_id}", fg="red"))
            return

        user.is_active = False
        db.session.commit()
        click.echo(click.style(f"✓ Deactivated user: {user.username}", fg="green"))


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
def reset_db():
    """Reset database (drop and recreate all tables)."""
    with app.app_context():
        click.echo("Dropping all tables...")
        db.drop_all()
        click.echo("Creating all tables...")
        db.create_all()
        click.echo(click.style("✓ Database reset complete", fg="green"))


@cli.command()
def init_db():
    """Initialize database (create tables)."""
    with app.app_context():
        db.create_all()
        click.echo(click.style("✓ Database initialized", fg="green"))


@cli.command()
def stats():
    """Show system statistics."""
    with app.app_context():
        from sqlalchemy import func

        user_count = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        doc_count = Document.query.count()
        query_count = QueryLog.query.count()
        api_key_count = APIKey.query.count()

        click.echo("\nSystem Statistics:")
        click.echo("-" * 40)
        click.echo(f"Users:          {user_count} ({active_users} active)")
        click.echo(f"Documents:      {doc_count}")
        click.echo(f"Queries:        {query_count}")
        click.echo(f"API Keys:       {api_key_count}")
        click.echo("-" * 40 + "\n")


@cli.command()
@click.argument("username")
def create_api_key(username):
    """Create API key for a user."""
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        if not user:
            click.echo(click.style(f"✗ User not found: {username}", fg="red"))
            return

        name = click.prompt("API Key name")
        description = click.prompt("Description (optional)", default="")
        permissions = click.prompt(
            "Permissions (comma-separated)",
            default="read,write",
        ).split(",")

        success, result = AuthService.create_api_key(
            user_id=user.id,
            name=name,
            description=description,
            permissions=[p.strip() for p in permissions],
        )

        if success:
            click.echo(click.style("\n✓ API Key created successfully!", fg="green"))
            click.echo(click.style("\nIMPORTANT: Save this key securely!", fg="yellow"))
            click.echo(click.style(f"\nAPI Key: {result['key']}\n", fg="cyan"))
        else:
            click.echo(click.style(f"✗ Failed: {result.get('error')}", fg="red"))


if __name__ == "__main__":
    cli()
