# 最终运行文件
import os.path
from flask import Flask, render_template, request, Response, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import video_recognize as vr
import single_image_recoginize as sr
import config
from hashlib import md5


app = Flask(__name__)
app.config.from_object(config)
db = SQLAlchemy(app)

final_path = ''
path_dir = []


def password_to_md5(s):
    new_md5 = md5()
    new_md5.update(s.encode(encoding='utf-8'))
    return new_md5.hexdigest()


# 数据库建表
class User(db.Model):
    __tablename__ = 'userinfo'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True)
    password = db.Column(db.String(120), unique=False)


def allowed_file_s(filename):
    return '.' in filename and os.path.splitext(filename)[1] in app.config['ALLOWED_EXTENSIONS_1']


def allowed_file_d(filename):
    return '.' in filename and os.path.splitext(filename)[1] in app.config['ALLOWED_EXTENSIONS_2']


# 登陆界面
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['userinput']
        password = request.form['passwordinput']
        user = User.query.filter_by(username=username, password=password_to_md5(password)).first()
        if user:
            session['username'] = username
            session['password'] = password_to_md5(password)
            return redirect('/main_page.html', code=301)
        else:
            user = User.query.filter_by(username=username).first()
            if user:
                return render_template('login.html', error='密码错误')
            return render_template('login.html', error='用户不存在，请注册')
    else:
        return render_template('login.html', error="")


# 注册界面
@app.route('/register.html', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['userinput']
        password = request.form['passwordinput']
        user = User.query.filter_by(username=username).all()
        if user:
            return render_template('register.html', error='用户名已存在')
        else:
            user1 = User(username=username, password=password_to_md5(password))
            db.session.add(user1)
            db.session.commit()
            return redirect('/main_page.html', code=301)
    else:
        return render_template('register.html', error="")


# 主页面
@app.route('/')
def showmainpage():
    return render_template('main_page.html')


# 产品介绍页面
@app.route('/technology.html')
def technology():
    return render_template('technology.html')


# 成员介绍页面
@app.route('/member.html')
def member():
    return render_template('member.html')


# 主页面，实现点击导航栏跳到主页面功能
@app.route('/main_page.html')
def main_page():
    return render_template('main_page.html')


# 联系我们页面
@app.route('/contactus.html')
def contact():
    return render_template('contactus.html')


# 动态识别页面
@app.route('/using.html', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        form = request.form['formid']
        if form == '0':
            f = request.files['upload']
            if f and allowed_file_d(f.filename):
                uploadpath = os.path.dirname(os.path.realpath(__file__))
                final_path = os.path.join(uploadpath, 'static/upload', f.filename)
                f.save(final_path)
                information = '文件上传成功'
                path_dir.append(f.filename)
                return render_template('recognize.html', information=information, filepath='upload/' + f.filename)
            else:
                return render_template('using.html', error='暂不支持此类文件')
        else:
            name = session.get('username')
            if not name:
                return "请登录"
            else:
                path = 'upload/' + path_dir[len(path_dir) - 1]
                f, r = vr.main(video_paths='static/' + path)
                return render_template('resultpage.html', filepath=path, frame=f, result=r)
    else:
        return render_template('using.html')


# 选择功能页面
@app.route('/choose.html')
def choose():
    return render_template('choose.html')


# 静态识别页面
@app.route('/staticrec.html', methods=['GET', 'POST'])
def staticrec():
    if request.method == 'POST':
        form = request.form['formid']
        if form == '0':
            f = request.files['upload']
            if f and allowed_file_s(f.filename):
                uploadpath = os.path.dirname(os.path.realpath(__file__))
                final_path = os.path.join(uploadpath, 'static/upload', f.filename)
                f.save(final_path)
                information = '文件上传成功'
                path_dir.append(f.filename)
                return render_template('recognize_static.html', information=information,
                                       filepath='upload/' + f.filename)
            else:
                return render_template('using_static.html', error='暂不支持此类文件')
        else:
            name = session.get('username')
            if not name:
                return "请登录"
            else:
                path = 'upload/' + path_dir[len(path_dir) - 1]
                r = sr.main(file_path='static/' + path)
                return render_template('resultpage_static.html', filepath=path, result=r)
    else:
        return render_template('using_static.html')


if __name__ == '__main__':
    # db.create_all()
    app.run()
