# coding:utf8

from app import app
"""
from flask_script import Manager, Server

manage = Manager(app)
manage.add_command("runserver", Server(
    host="127.0.0.1", port=5000)
)
"""
# 项目入口
if __name__ == "__main__":
    app.run()
