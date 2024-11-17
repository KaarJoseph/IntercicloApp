import pycuda.autoinit
from flask import Flask, request, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from PIL import Image
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Configuración básica
app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Configuración de la base de datos PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://FilterApp:FilterApp@localhost/FilterApp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Directorios de archivos
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Modelo de Usuario
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

# Modelo de Publicación
class Post(db.Model):
    __tablename__ = 'posts'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    processed_image_path = db.Column(db.String(200), nullable=True)
    description = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# Crear tablas
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
        image = Image.open(image_path).convert('L')
        image_array = np.array(image, dtype=np.uint8)
        height, width = image_array.shape

        output_array = np.zeros_like(image_array)

        block_dim = (threads, threads, 1)
        grid_dim = ((width + block_dim[0] - 1) // block_dim[0], 
                    (height + block_dim[1] - 1) // block_dim[1], 1)

        image_gpu = cuda.mem_alloc(image_array.nbytes)
        output_gpu = cuda.mem_alloc(output_array.nbytes)

        cuda.memcpy_htod(image_gpu, image_array)
        kernel = compiled_kernels[filter_name].get_function(f"{filter_name}_filter")
        kernel(image_gpu, output_gpu, np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)
        cuda.memcpy_dtoh(output_array, output_gpu)

        output_image = Image.fromarray(output_array)
        output_path = os.path.join(RESULT_FOLDER, f"filtered_{filter_name}.png")
        output_image.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error applying filter: {e}")
        return None

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    hashed_password = generate_password_hash(data['password'], method='sha256')
    new_user = User(username=data['username'], email=data['email'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "Usuario registrado exitosamente"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if user and check_password_hash(user.password, data['password']):
        session['user_id'] = user.id
        return jsonify({"message": "Inicio de sesión exitoso"}), 200
    return jsonify({"error": "Credenciales inválidas"}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({"message": "Sesión cerrada exitosamente"}), 200

@app.route('/posts', methods=['POST', 'GET'])
def manage_posts():
    if request.method == 'POST':
        if 'user_id' not in session:
            return jsonify({"error": "Debe iniciar sesión"}), 401
        data = request.form
        image = request.files['image']
        description = data.get('description', '')

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        new_post = Post(user_id=session['user_id'], image_path=image_path, description=description)
        db.session.add(new_post)
        db.session.commit()

        return jsonify({"message": "Publicación creada exitosamente"}), 201

    elif request.method == 'GET':
        posts = Post.query.all()
        return jsonify([{
            "id": post.id,
            "user_id": post.user_id,
            "description": post.description,
            "image_path": post.image_path,
            "processed_image_path": post.processed_image_path
        } for post in posts]), 200

@app.route('/posts/<int:post_id>/filter', methods=['POST'])
def apply_post_filter(post_id):
    if 'user_id' not in session:
        return jsonify({"error": "Debe iniciar sesión"}), 401

    post = Post.query.get(post_id)
    if not post or post.user_id != session['user_id']:
        return jsonify({"error": "Publicación no encontrada o no autorizada"}), 404

    filter_name = request.json.get('filter_name')
    threads = int(request.json.get('threads', 4))

    if filter_name not in kernels:
        return jsonify({"error": "Filtro no válido"}), 400

    filtered_image_path = apply_filter(post.image_path, filter_name, threads)
    if not filtered_image_path:
        return jsonify({"error": "Error al aplicar filtro"}), 500

    post.processed_image_path = filtered_image_path
    db.session.commit()

    return jsonify({"message": "Filtro aplicado exitosamente", "filtered_image_path": filtered_image_path}), 200

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
