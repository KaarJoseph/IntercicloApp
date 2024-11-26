import pycuda.driver as cuda
import pycuda.autoinit
from flask import Flask, request, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from PIL import Image
import numpy as np
from pycuda.compiler import SourceModule
from datetime import datetime, timedelta
import os
import uuid
from flask_session import Session

# Configuración básica
app = Flask(__name__)
app.secret_key = 'super_secret_key'
CORS(app, supports_credentials=True)  # Permite cookies con solicitudes CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuración de la base de datos PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://FilterApp:FilterApp@localhost/FilterApp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Directorios de archivos
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inicialización de contexto de CUDA
cuda.init()
device = cuda.Device(0)
cuda_context = device.make_context()

# Configuración de Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'  # Almacena la sesión en el sistema de archivos
app.config['SESSION_PERMANENT'] = False  # La sesión no es permanente
app.config['SESSION_COOKIE_HTTPONLY'] = True  # La cookie solo es accesible por HTTP
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Protege contra CSRF
app.config['SESSION_COOKIE_SECURE'] = False  # Cambia a True si usas HTTPS
Session(app)

# ------------------ Modelos ------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    profile_picture = db.Column(db.String(200), nullable=True)
    description = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Post(db.Model):
    __tablename__ = 'posts'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    processed_image_path = db.Column(db.String(200), nullable=True)
    description = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('posts', lazy=True))

class Like(db.Model):
    __tablename__ = 'likes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('posts.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Comment(db.Model):
    __tablename__ = 'comments'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('posts.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Share(db.Model):
    __tablename__ = 'shares'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('posts.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    __tablename__ = 'notifications'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    related_id = db.Column(db.Integer, nullable=False)
    action = db.Column(db.String(50), nullable=False)
    related_type = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Story(db.Model):
    __tablename__ = 'stories'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    media_url = db.Column(db.String(200), nullable=False)
    text = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    user = db.relationship('User', backref=db.backref('stories', lazy=True))

class Follow(db.Model):
    __tablename__ = 'follows'
    id = db.Column(db.Integer, primary_key=True)
    follower_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    followed_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    follower = db.relationship('User', foreign_keys=[follower_id], backref='following')
    followed = db.relationship('User', foreign_keys=[followed_id], backref='followers')


with app.app_context():
    db.create_all()

# CUDA Kernels
kernels = {
    "sobel": """
        __global__ void sobel_filter(unsigned char *image, unsigned char *output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int gx = -image[(y - 1) * width + (x - 1)] - 2 * image[y * width + (x - 1)] - image[(y + 1) * width + (x - 1)]
                         + image[(y - 1) * width + (x + 1)] + 2 * image[y * width + (x + 1)] + image[(y + 1) * width + (x + 1)];
                int gy = -image[(y - 1) * width + (x - 1)] - 2 * image[(y - 1) * width + x] - image[(y - 1) * width + (x + 1)]
                         + image[(y + 1) * width + (x - 1)] + 2 * image[(y + 1) * width + x] + image[(y + 1) * width + (x + 1)];
                int magnitude = min(255, (int)sqrtf(gx * gx + gy * gy));
                output[y * width + x] = magnitude;
            }
        }
    """,
    "erosion": """
        __global__ void erosion_filter(unsigned char *image, unsigned char *output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int min_val = 255;
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        min_val = min(min_val, (int)image[(y + dy) * width + (x + dx)]);
                    }
                }
                output[y * width + x] = min_val;
            }
        }
    """,
    "highpass": """
        __global__ void highpass_filter(unsigned char *image, unsigned char *output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int center = image[y * width + x];
                int avg_neighbors = (image[(y - 1) * width + x] + image[(y + 1) * width + x] +
                                     image[y * width + (x - 1)] + image[y * width + (x + 1)]) / 4;
                output[y * width + x] = max(0, min(255, center - avg_neighbors));
            }
        }
    """,
    "gaussian": """
        __global__ void gaussian_filter(unsigned char *image, unsigned char *output, int width, int height) {
            int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int sum = 0;
                int weight = 0;
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        int val = image[(y + dy) * width + (x + dx)];
                        int w = kernel[dy + 1][dx + 1];
                        sum += val * w;
                        weight += w;
                    }
                }
                output[y * width + x] = sum / weight;
            }
        }
    """,
    "emboss": """
        __global__ void emboss_filter(unsigned char *image, unsigned char *output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int val = image[y * width + x] - image[(y + 1) * width + (x + 1)] + 128;
                output[y * width + x] = max(0, min(255, val));
            }
        }
    """
}

compiled_kernels = {name: SourceModule(kernel) for name, kernel in kernels.items()}

def apply_filter(image_path, filter_name, threads):
    try:
        # Activar el contexto de CUDA
        cuda_context.push()

        # Cargar la imagen y convertirla a escala de grises
        image = Image.open(image_path).convert('L')
        image_array = np.array(image, dtype=np.uint8)
        height, width = image_array.shape

        # Preparar la salida
        output_array = np.zeros_like(image_array)

        # Configuración de CUDA
        block_dim = (threads, threads, 1)
        grid_dim = ((width + block_dim[0] - 1) // block_dim[0],
                    (height + block_dim[1] - 1) // block_dim[1], 1)

        # Reservar memoria en la GPU
        image_gpu = cuda.mem_alloc(image_array.nbytes)
        output_gpu = cuda.mem_alloc(output_array.nbytes)

        # Copiar datos a la GPU
        cuda.memcpy_htod(image_gpu, image_array)

        # Ejecutar el kernel CUDA
        kernel = compiled_kernels[filter_name].get_function(f"{filter_name}_filter")
        kernel(image_gpu, output_gpu, np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)

        # Recuperar los datos procesados de la GPU
        cuda.memcpy_dtoh(output_array, output_gpu)

        # Guardar la imagen procesada
        output_image = Image.fromarray(output_array)
        output_path = os.path.join(RESULT_FOLDER, f"filtered_{filter_name}.png")
        output_image.save(output_path)

        return output_path
    except Exception as e:
        print(f"Error en apply_filter: {e}")
        return None
    finally:
        # Liberar el contexto
        cuda_context.pop()

# Endpoints principales
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    new_user = User(username=data['username'], email=data['email'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "Usuario registrado exitosamente"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if user and check_password_hash(user.password, data['password']):
        session['user_id'] = user.id  # Asegúrate de que se guarda correctamente el user_id
        print(f"Sesión iniciada: Usuario ID {user.id}")  # Depuración
        return jsonify({
            "message": "Inicio de sesión exitoso",
            "user_id": user.id
        }), 200
    return jsonify({"error": "Credenciales inválidas"}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)  # Elimina el user_id de la sesión
    return jsonify({"message": "Sesión cerrada exitosamente"}), 200

@app.route('/posts', methods=['POST', 'GET'])
def manage_posts():
    if request.method == 'POST':
        try:
            if 'user_id' not in session:
                return jsonify({"error": "Debe iniciar sesión"}), 401

            user_id = session['user_id']
            description = request.form.get('description', '')

            # Verifica que se envió una imagen
            image = request.files.get('image')
            if not image:
                return jsonify({"error": "Debe incluirse una imagen"}), 400

            # Guarda la imagen en el directorio de uploads
            filename = secure_filename(image.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(image_path)

            # Crea la publicación en la base de datos
            new_post = Post(
                user_id=user_id,
                image_path=image_path,
                description=description
            )
            db.session.add(new_post)
            db.session.commit()

            return jsonify({
                "message": "Publicación creada",
                "id": new_post.id
            }), 201

        except Exception as e:
            print(f"Error en el endpoint POST /posts: {e}")
            return jsonify({"error": "Error interno del servidor"}), 500

    elif request.method == 'GET':
        try:
            # Obtener todas las publicaciones
            posts = Post.query.order_by(Post.created_at.desc()).all()
            response = []

            for post in posts:
                user = User.query.get(post.user_id)
                response.append({
                    "post_id": post.id,
                    "user_id": post.user_id,
                    "username": user.username if user else "Usuario desconocido",
                    "profile_picture": user.profile_picture if user else None,
                    "image_path": f"http://{request.host}/{post.image_path.replace('\\', '/')}",
                    "description": post.description,
                    "created_at": post.created_at.isoformat(),
                })

            return jsonify(response), 200

        except Exception as e:
            print(f"Error en el endpoint GET /posts: {e}")
            return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/posts/<int:post_id>/filter', methods=['POST'])
def apply_post_filter(post_id):
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Debe iniciar sesión"}), 401

        post = Post.query.get(post_id)
        if not post or post.user_id != session['user_id']:
            return jsonify({"error": "Publicación no encontrada o no autorizada"}), 404

        data = request.get_json()
        filter_name = data.get('filter_name')
        threads = int(data.get('threads', 4))

        if not filter_name or filter_name not in kernels:
            return jsonify({"error": "Filtro no válido"}), 400

        # Generar un nombre único para la imagen procesada
        filtered_filename = f"{uuid.uuid4().hex}_filtered_{filter_name}.png"
        filtered_filepath = os.path.join(RESULT_FOLDER, filtered_filename)

        # Aplicar el filtro
        filtered_image_path = apply_filter(post.image_path, filter_name, threads)
        if not filtered_image_path:
            return jsonify({"error": "Error al aplicar filtro"}), 500

        # Renombrar la imagen procesada con el nuevo nombre único
        os.rename(filtered_image_path, filtered_filepath)

        post.processed_image_path = filtered_filepath
        db.session.commit()

        return jsonify({
            "message": "Filtro aplicado exitosamente",
            "filtered_image_path": filtered_filepath
        }), 200

    except Exception as e:
        print(f"Error en apply_post_filter: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500


@app.route('/posts/<int:post_id>/like', methods=['POST'])
def like_post(post_id):
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401
    user_id = session['user_id']
    existing_like = Like.query.filter_by(user_id=user_id, post_id=post_id).first()
    if existing_like:
        db.session.delete(existing_like)
        db.session.commit()
        return jsonify({"message": "Like eliminado"}), 200
    new_like = Like(user_id=user_id, post_id=post_id)
    db.session.add(new_like)
    # Crear notificación
    new_notification = Notification(user_id=session['user_id'], post_id=post_id, action='like')
    db.session.add(new_notification)
    db.session.commit()
    return jsonify({"message": "Like agregado"}), 201

@app.route('/posts/<int:post_id>/likes', methods=['GET'])
def get_post_likes(post_id):
    likes = Like.query.filter_by(post_id=post_id).all()
    if not likes:
        return jsonify({"message": "No hay likes para esta publicación"}), 200
    users = [{"user_id": like.user_id} for like in likes]
    return jsonify({"likes": users}), 200

@app.route('/posts/<int:post_id>/comment', methods=['POST'])
def comment_post(post_id):
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401
    content = request.json.get('content')
    if not content:
        return jsonify({"error": "El comentario no puede estar vacío"}), 400
    new_comment = Comment(user_id=session['user_id'], post_id=post_id, content=content)
    db.session.add(new_comment)
    db.session.commit()
    # Crear notificación
    new_notification = Notification(user_id=session['user_id'], post_id=post_id, action='comment')
    db.session.add(new_notification)
    db.session.commit()
    return jsonify({"message": "Comentario agregado"}), 201

@app.route('/posts/<int:post_id>/comments', methods=['GET'])
def get_post_comments(post_id):
    comments = Comment.query.filter_by(post_id=post_id).all()
    if not comments:
        return jsonify({"message": "No hay comentarios para esta publicación"}), 200
    response = [{"user_id": comment.user_id, "content": comment.content, "created_at": comment.created_at} for comment in comments]
    return jsonify({"comments": response}), 200

@app.route('/posts/<int:post_id>/share', methods=['POST'])
def share_post(post_id):
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401
    new_share = Share(user_id=session['user_id'], post_id=post_id)
    db.session.add(new_share)
    db.session.commit()
    return jsonify({"message": "Publicación compartida"}), 201

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

    # ------------------ Nuevas Funcionalidades ------------------

@app.route('/users/<int:user_id>/posts', methods=['GET'])
def get_user_posts(user_id):
    try:
        posts = Post.query.filter_by(user_id=user_id).all()
        return jsonify([{
            "id": post.id,
            "description": post.description,
            "image_path": post.processed_image_path or post.image_path
        } for post in posts]), 200
    except Exception as e:
        print(f"Error en get_user_posts: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500

# Endpoint para editar perfil
@app.route('/profile/edit', methods=['POST'])
def edit_profile():
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401

    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({"error": "Usuario no encontrado"}), 404

    data = request.form  # Cambia a form para manejar texto e imágenes
    user.username = data.get('username', user.username)
    user.description = data.get('description', user.description)

    if 'profile_picture' in request.files:
        picture = request.files['profile_picture']
        picture_path = os.path.join(UPLOAD_FOLDER, secure_filename(picture.filename))
        picture.save(picture_path)
        user.profile_picture = f"/uploads/{secure_filename(picture.filename)}"  # Usar URL relativa

    db.session.commit()
    return jsonify({"message": "Perfil actualizado exitosamente"}), 200

# Endpoint para buscar usuarios
@app.route('/users/search', methods=['GET'])
def search_users():
    query = request.args.get('query', '').lower()
    users = User.query.filter(
        (User.username.ilike(f"%{query}%")) | (User.email.ilike(f"%{query}%"))
    ).all()
    
    return jsonify([{
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "profile_picture": user.profile_picture,
        "description": user.description
    } for user in users]), 200

# Endpoint para crear una historia
@app.route('/stories', methods=['POST'])
def create_story():
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401

    user_id = session['user_id']
    media = request.files.get('media')
    text = request.form.get('text', '')
    expiration_minutes = int(request.form.get('expiration_minutes', 1440))  # Default: 24 hours

    if not media:
        return jsonify({"error": "Debe subir un archivo multimedia"}), 400

    media_path = os.path.join(UPLOAD_FOLDER, secure_filename(media.filename))
    media.save(media_path)

    expires_at = datetime.utcnow() + timedelta(minutes=expiration_minutes)
    new_story = Story(user_id=user_id, media_url=media_path, text=text, expires_at=expires_at)
    db.session.add(new_story)
    db.session.commit()

    return jsonify({"message": "Historia creada exitosamente"}), 201

# Endpoint para obtener historias activas
@app.route('/stories/active', methods=['GET'])
def get_active_stories():
    current_time = datetime.utcnow()
    stories = Story.query.filter(Story.expires_at > current_time).all()

    return jsonify([{
        "id": story.id,
        "user_id": story.user_id,
        "media_url": story.media_url,
        "text": story.text,
        "created_at": story.created_at,
        "expires_at": story.expires_at
    } for story in stories]), 200

# Limpieza automática de historias expiradas
@app.route('/stories/cleanup', methods=['POST'])
def cleanup_stories():
    current_time = datetime.utcnow()
    expired_stories = Story.query.filter(Story.expires_at <= current_time).all()
    
    for story in expired_stories:
        db.session.delete(story)
    
    db.session.commit()
    return jsonify({"message": f"Se eliminaron {len(expired_stories)} historias expiradas"}), 200

# Notificaciones extendidas
@app.route('/notifications', methods=['GET'])
def get_notifications():
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401
    
    user_id = session['user_id']
    notifications = Notification.query.filter_by(user_id=user_id).all()
    
    return jsonify([{
        "id": notif.id,
        "action": notif.action,
        "related_id": notif.related_id,
        "related_type": notif.related_type,
        "created_at": notif.created_at
    } for notif in notifications]), 200

# Lógica para notificar visualización de historias
@app.route('/stories/<int:story_id>/view', methods=['POST'])
def view_story(story_id):
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401

    story = Story.query.get(story_id)
    if not story:
        return jsonify({"error": "Historia no encontrada"}), 404

    if story.expires_at <= datetime.utcnow():
        return jsonify({"error": "Esta historia ya expiró"}), 400

    # Crear notificación de visualización
    new_notification = Notification(
        user_id=story.user_id,
        related_id=story.id,
        action="view_story",
        related_type="story"
    )
    db.session.add(new_notification)
    db.session.commit()

    return jsonify({"message": "Historia visualizada exitosamente"}), 200

@app.route('/users/<int:user_id>/follow', methods=['POST'])
def follow_user(user_id):
    try:
        data = request.get_json()  # Asegúrate de que sea JSON
        current_user_id = data.get('current_user_id')

        if not current_user_id:
            return jsonify({"error": "User ID is required"}), 400

        if current_user_id == user_id:
            return jsonify({"error": "Cannot follow yourself"}), 400

        existing_follow = Follow.query.filter_by(follower_id=current_user_id, followed_id=user_id).first()
        if existing_follow:
            db.session.delete(existing_follow)
            db.session.commit()
            return jsonify({"message": "Unfollowed successfully"}), 200

        new_follow = Follow(follower_id=current_user_id, followed_id=user_id)
        db.session.add(new_follow)
        db.session.commit()
        return jsonify({"message": "Followed successfully"}), 201

    except Exception as e:
        print(f"Error in follow_user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/users/<int:user_id>/followers', methods=['GET'])
def get_followers(user_id):
    followers = Follow.query.filter_by(followed_id=user_id).all()
    return jsonify([{
        "id": follower.follower.id,
        "username": follower.follower.username,
        "profile_picture": follower.follower.profile_picture or ""
    } for follower in followers]), 200

@app.route('/users/<int:user_id>/following', methods=['GET'])
def get_following(user_id):
    following = Follow.query.filter_by(follower_id=user_id).all()
    return jsonify([{
        "id": followed.followed.id,
        "username": followed.followed.username,
        "profile_picture": followed.followed.profile_picture or ""
    } for followed in following]), 200

@app.route('/profile', methods=['GET'])
def get_profile():
    if 'user_id' not in session:
        print("Sesión inválida o expirada.")  # Log para depurar
        return jsonify({"error": "Debes iniciar sesión"}), 401

    user = User.query.get(session['user_id'])
    if not user:
        print("Usuario no encontrado en la base de datos.")  # Log para depurar
        return jsonify({"error": "Usuario no encontrado"}), 404

    followers_count = Follow.query.filter_by(followed_id=user.id).count()
    following_count = Follow.query.filter_by(follower_id=user.id).count()

    print(f"Usuario logueado: {user.username}, ID: {user.id}")  # Log para depurar
    return jsonify({
        "username": user.username,
        "description": user.description or "Sin descripción",
        "profile_picture": user.profile_picture or "",
        "publicaciones": len(user.posts),
        "seguidores": followers_count,
        "seguidos": following_count,
        "posts": [post.image_path for post in user.posts]
    }), 200

@app.before_request
def check_cookies():
    print("Cookies recibidas:", request.cookies)

@app.route('/uploads/<path:filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/home_posts', methods=['GET'])
def get_home_posts():
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401

    posts = Post.query.order_by(Post.created_at.desc()).all()
    response = []
    for post in posts:
        user = User.query.get(post.user_id)
        response.append({
            "post_id": post.id,
            "user_id": post.user_id,
            "username": user.username,
            "profile_picture": user.profile_picture,
            "image_path": f"http://{request.host}/{post.image_path.replace('\\', '/')}",  # Corrige \ a /
            "description": post.description,
            "created_at": post.created_at
        })
    return jsonify(response), 200

@app.route('/posts', methods=['POST'])
def create_post():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Debe iniciar sesión"}), 401

        user_id = session['user_id']
        description = request.form.get('description')
        image = request.files.get('image')

        if not image:
            return jsonify({"error": "No se proporcionó ninguna imagen"}), 400

        # Generar un nombre único para la imagen
        filename = f"{uuid.uuid4().hex}_{secure_filename(image.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Guardar la imagen en el servidor
        image.save(filepath)

        # Crear el post en la base de datos
        post = Post(user_id=user_id, description=description, image_path=filepath)
        db.session.add(post)
        db.session.commit()

        return jsonify({"message": "Publicación creada exitosamente", "id": post.id}), 201

    except Exception as e:
        print(f"Error en create_post: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.before_request
def log_session():
    print(f"Cookies recibidas: {request.cookies}")
    print(f"Session user_id: {session.get('user_id')}")

@app.route('/results/<path:filename>', methods=['GET'])
def get_filtered_image(filename):
    return send_from_directory('results', filename)

@app.route('/posts/<int:post_id>/finalize', methods=['POST'])
def finalize_post(post_id):
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Debe iniciar sesión"}), 401

        post = Post.query.get(post_id)
        if not post or post.user_id != session['user_id']:
            return jsonify({"error": "Publicación no encontrada o no autorizada"}), 404

        data = request.get_json()
        processed_image_path = data.get('processed_image_path')
        if not processed_image_path:
            return jsonify({"error": "No se proporcionó la imagen procesada"}), 400

        post.processed_image_path = processed_image_path
        db.session.commit()

        return jsonify({"message": "Publicación actualizada exitosamente"}), 200
    except Exception as e:
        print(f"Error en finalize_post: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
