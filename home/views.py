# coding:utf8

from app import db, app
from app.home import home
from app.home.forms import RegistForm, LoginForm, UserdetailForm, PwdForm, CheckForm
from app.models import User
from flask import render_template, redirect, url_for, flash, session, request, make_response,jsonify
from werkzeug.security import generate_password_hash
from functools import wraps
import uuid, os, datetime
import base64
import pickle
from app.PCASVM_ACTION import judge
from app.GetImage import GetImage
from app.PCASVMTRY import trainPCA
#from app.face_reco import prepare_data,judgeFace,person_data
from app.keras_face import getImageKeras
from app.keras_train import trainKeras
import shutil
# 定义登录判断装饰器
def user_login_req(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # session不存在时请求登录
        if "user" not in session:
            return redirect(url_for("home.login", next=request.url))
        return f(*args, **kwargs)

    return decorated_function


# 修改文件名称
def change_filename(filename):
    fileinfo = os.path.splitext(filename)  # 对名字进行前后缀分离
    filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex + fileinfo[-1]  # 生成新文件名
    return filename


# 定义首页列表视图
@home.route("/", methods=["GET", "POST"])
def index():
    form = CheckForm()  # 导入登录表单
    #faceHistograms = []
    if form.validate_on_submit():  # 验证是否有提交表单
        data = form.data
        user = User.query.filter_by(phone=data["phone"]).first()
        # 读取图片并存入check文件夹
        img = request.form.get("face")
        code = img.replace('data:image/png;base64,','')
        code2 = "\"" + code+ "\""
        code2=base64.b64decode(code2)
        if not os.path.exists(app.config['UP_DIR'] + "check" + os.sep):
            os.makedirs(app.config['UP_DIR'] + "check" + os.sep)
        filename = app.config['UP_DIR'] +"check" +os.sep +str(user.id) +".jpg"
        filename2 = "F:/biyesheji/PythonStudyCode/app/static/uploads/check/"+str(user.id)+".jpg"
        file = open(filename,'wb')
        file.write(code2)
        file.close()

        images_names = os.listdir("F:/biyesheji/PythonStudyCode/app/static/uploads/users")
        print(filename2)
        #checker_jpg = judgeFace(filename2, faceHistograms, images_names)
        if len(images_names) != 0:
            checker_jpg = judge(filename2)
            print(checker_jpg)
            checker_id = checker_jpg.replace('].jpg', '')
            checker_id = int(checker_id.replace('[',''))
            checker = User.query.filter_by(id=checker_id).first()
            if checker.id == user.id:
                user.daka = 1
            else:
                user.daka = 0
            db.session.add(user)
            db.session.commit()
    else:
        user = ""
        checker = ""
    return render_template("home/index.html", form=form, user=user, checker=checker)


# 定义注册视图
@home.route("/regist/", methods=["GET", "POST"])
def regist():
    form = RegistForm()
    if form.validate_on_submit():
        data = form.data
        user = User(
            name=data["name"],
            pwd=generate_password_hash(data["pwd"]),
            email=data["email"],
            phone=data["phone"],
            school=data["school"],
            uuid=uuid.uuid4().hex
        )
        db.session.add(user)
        db.session.commit()
        flash("注册成功，请登录！", "ok")
        return redirect(url_for("home.login"))
    return render_template("home/regist.html", form=form)


# 定义登录视图
@home.route("/login/", methods=["GET", "POST"])
def login():
    form = LoginForm()  # 导入登录表单
    if form.validate_on_submit():  # 验证是否有提交表单
        data = form.data
        user = User.query.filter_by(phone=data["phone"]).first()
        if not user.check_pwd(data["pwd"]):
            flash("密码错误！", "err")
            return redirect(url_for("home.login"))
        session["user"] = data["phone"]
        session["user_id"] = user.id
        return redirect(request.args.get("next") or url_for("home.user"))
    return render_template("home/login.html", form=form)


# 定义登出视图
@home.route("/logout/")
@user_login_req
def logout():
    session.pop("user")
    session.pop("user_id")
    return redirect(url_for("home.login"))


# 定义学生中心视图
@home.route("/user/", methods=["GET", "POST"])
@user_login_req
def user():
    form = UserdetailForm()
    user = User.query.get(int(session["user_id"]))
    if user.face is not None:
        form.face.validators = []
    if request.method == "GET":
        form.name.data = user.name
        form.email.data = user.email
        form.phone.data = user.phone
        form.school.data = user.school
    if form.validate_on_submit():
        data = form.data

        if not os.path.exists(app.config['UP_DIR'] + "users" + os.sep):
            os.makedirs(app.config['UP_DIR'] + "users" + os.sep)

        if not os.path.exists(app.config['UP_DIR'] + "users" + os.sep+str(user.id)+os.sep):
            os.makedirs(app.config['UP_DIR'] + "users" + os.sep+str(user.id)+os.sep)

        if form.face.data.filename != '':
            old_face = user.face
            if old_face is not None and os.path.exists(app.config['UP_DIR'] + "users" + os.sep + old_face + ".jpg"):
                os.remove(app.config['UP_DIR'] + "users" + os.sep + old_face + ".jpg")
            user.face = form.face
            user.face = str(user.id)

            form.face.data.save(app.config['UP_DIR'] + "users" + os.sep + str(user.id)+ os.sep + "1.jpg")
            #user.tezheng, rect = person_data("F:/Identification_System/app/static/uploads/users", user.id)

        user.name = data["name"]
        user.email = data["email"]
        user.phone = data["phone"]
        user.school = data["school"]
        db.session.add(user)
        db.session.commit()
        flash("修改成功！", "ok")
    return render_template("home/user.html", form=form, user=user)


# 定义修改密码视图
@home.route("/pwd/", methods=["GET", "POST"])
@user_login_req
def pwd():
    form = PwdForm()
    if form.validate_on_submit():
        data = form.data
        user = User.query.filter_by(name=session["user"]).first()
        user.pwd = generate_password_hash(data["new_pwd"])
        db.session.add(user)
        db.session.commit()
        flash("修改密码成功，请重新登录！", "ok")
        return redirect(url_for("home.logout"))
    return render_template("home/pwd.html", form=form)


ALLOWED_EXTENSIONS = set(['png','jpg','JPG','PNG','bmp'])


def allowed_file(filename):
    return '.'in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS


# 定义上传照片视图
@home.route("/upload/", methods=["GET", "POST"])
@user_login_req
def upload():
    names=['0']*100
    user = User.query.filter_by(phone=session["user"]).first()
    if user.index >=1:
        if user.index < 10:
            lenth = user.index
        else:
            lenth = 10
        for i in range(1, lenth + 1):
            names[i] = str(i)
    else:
        names = ""
        lenth = 0
    if request.method == 'POST':
        upload_file = request.files.getlist("file[]")
        filenames = []
        for f in upload_file:
            if not(f and allowed_file(f.filename)):
                return jsonify({"error":1001,"msg":"请检查上传的图片类型，仅限于png/PNG/jpg/JPG/bmp"})


            if not os.path.exists(app.config['UP_DIR'] + "users" + os.sep):
                os.makedirs(app.config['UP_DIR'] + "users" + os.sep)

            if not os.path.exists(app.config['UP_DIR'] + "users" + os.sep+str(user.id)+os.sep):
                os.makedirs(app.config['UP_DIR'] + "users" + os.sep+str(user.id)+os.sep)
            user.index = user.index+1
            upload_path = app.config['UP_DIR'] + "users" + os.sep + str(user.id) + os.sep + str(user.index)+".jpg"
            f.save(upload_path)
            filenames.append(user.index)
            db.session.add(user)
            db.session.commit()

        return render_template('home/upload.html',names = names,user=user,lenth=lenth,filenames = filenames)
    return render_template('home/upload.html', names=names, user=user, lenth=lenth)


@home.route("/fasttrain/", methods=["GET", "POST"])
def fasttrain():
    print("Preparing data...")
    images_names = os.listdir("F:/Identification_System/app/static/uploads/users")
    lenth= len(images_names)
    if request.method == 'POST':
        string1 = "正在加载图片"
        GetImage()
        string2 = "正在训练数据"
        report,t0 = trainPCA()
        string3 ="训练结束"
        return render_template('home/fasttrain.html',lenth=lenth,string1=string1,string2=string2,string3=string3,report=report,flag=1,t0=t0)
    return render_template('home/fasttrain.html',lenth=lenth,flag=0,t0=0)


@home.route("/precisetrain/", methods=["GET", "POST"])
def precisetrain():
    print("Preparing data...")
    #if os.path.exists("F:/Identification_System/app/static/uploads/users_dup"):
    #    shutil.rmtree("F:/Identification_System/app/static/uploads/users_dup")

    #shutil.copytree("F:/Identification_System/app/static/uploads/users", "F:/Identification_System/app/static/uploads/users_dup")
    images_names = os.listdir("F:/Identification_System/app/static/uploads/users_dup")
    lenth= len(images_names)
    if request.method == 'POST':
        string1 = "正在加载图片"
        getImageKeras()
        string2 = "正在训练数据,时间稍久，请耐心等待"
        report,t0 = trainKeras()
        string3 ="训练结束，训练精度为"
        return render_template('home/precisetrain.html',lenth=lenth,string1=string1,string2=string2,string3=string3,report=report,flag=1,t0=t0)
    return render_template('home/precisetrain.html',lenth=lenth,flag=0,t0=0)
