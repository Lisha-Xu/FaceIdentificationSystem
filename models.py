# coding:utf8
from app import db

'''
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
import pymysql
import os

app = Flask(__name__)  # 创建Flask对象

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456@127.0.0.1:3306/face"  # 定义Mysql数据库连接
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True  # 如果设置成 True (默认情况)，Flask-SQLAlchemy 将会追踪对象的修改并且发送信号
db = SQLAlchemy(app)  # 创建db对象
'''
# 会员
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)  # 编号
    name = db.Column(db.String(100), unique=True)  # 姓名
    pwd = db.Column(db.String(100))  # 密码
    email = db.Column(db.String(100), unique=True)  # 邮箱
    phone = db.Column(db.String(11), unique=True)  # 手机号码
    school = db.Column(db.String(100))  # 学校
    face = db.Column(db.String(255), unique=True)  # 头像
    index = db.Column(db.Integer, default=1)
    daka = db.Column(db.Integer)
    uuid = db.Column(db.String(255), unique=True)

    def __repr__(self):
        return "<User %r>" % self.name

    def check_pwd(self, pwd):
        from werkzeug.security import check_password_hash
        return check_password_hash(self.pwd, pwd)  # 验证密码是否正确，返回True和False


'''
# 将模型生成数据表
if __name__ == "__main__":
    db.drop_all('__all__')
    db.create_all()
    db.session.commit()
'''
