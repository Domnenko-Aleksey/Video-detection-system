import datetime
import sys
import base64
from cryptography import fernet
import jinja2
import aiohttp_jinja2
from aiohttp import web
from aiohttp_session import setup, get_session, session_middleware
from aiohttp_session.cookie_storage import EncryptedCookieStorage
from ultralytics import YOLO

sys.path.append('classes')
sys.path.append('components')
sys.path.append('components/auth')

from Core import Core
from auth import auth
from mainpage import mainpage
from video_upload_ajax import video_upload_ajax
from model_predict_ajax import model_predict_ajax


app = web.Application(client_max_size=1024**100)

# Sessions
fernet_key = fernet.Fernet.generate_key()
secret_key = base64.urlsafe_b64decode(fernet_key)
setup(app, EncryptedCookieStorage(secret_key))

# model = YOLO('yolov8l.pt')
model = YOLO('model/best.onnx', task='detect')
# model.predict('files/video.mp4', device=0)
# print('YOLO, names:', model.model.names)

CORE = Core()
CORE.debug_on = True
CORE.model = model

@aiohttp_jinja2.template('main.html')
async def index(request):
    now = datetime.datetime.now()
    CORE.debug(f'===== INDEX ===== {now}')
    CORE.debug('REQUEST:')
    CORE.debug(request)

    CORE.initial()
    CORE.procReq(request)
    CORE.post = await request.post()
    CORE.session = await get_session(request)

    # Auth
    if not auth(CORE):
        CORE.debug('Нет авторизации')
        return {'AUTH': False, 'content': CORE.content, 'head': CORE.getHead()}

    # Redirect auth
    if CORE.p[0] == 'auth':
        return web.HTTPFound('/')

    # Components
    functions = {
        '': mainpage.mainpage,
        'video_upload_ajax': video_upload_ajax.video_upload_ajax,
        'model_predict_ajax': model_predict_ajax.model_predict_ajax,
    }

    if (CORE.p[0] not in functions):
        raise web.HTTPNotFound()

    # functions
    r = functions[CORE.p[0]](CORE)

    # Redirect
    if r:
        if 'redirect' in r:
            return web.HTTPFound(r['redirect'])
        if 'ajax' in r:
            return web.HTTPOk(text=r['ajax'])

    return {'AUTH': True, 'content': CORE.content, 'head': CORE.getHead()}



aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))
app.add_routes([
    web.static('/templates', 'templates'),
    web.static('/files', 'files'),
    web.static('/runs', 'runs'),
    web.get('/{url:.*}', index),
    web.post('/{url:.*}', index),
])

if __name__ == '__main__':
    web.run_app(app, port=9100)