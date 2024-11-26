from app import app, db
from app import User, Post

# Crear datos de prueba
with app.app_context():
    # Elimina todas las tablas y vuelve a crearlas
    db.drop_all()
    db.create_all()

    # Crear usuarios de prueba
    user1 = User(username="Juan Pérez", email="juan@example.com", password="123456")
    user2 = User(username="Ana López", email="ana@example.com", password="123456")
    db.session.add(user1)
    db.session.add(user2)

    # Crear publicaciones de prueba
    post1 = Post(user_id=1, image_path="path/to/image1.png", description="Primera publicación")
    post2 = Post(user_id=2, image_path="path/to/image2.png", description="Segunda publicación")
    db.session.add(post1)
    db.session.add(post2)

    # Confirmar cambios
    db.session.commit()

    print("Datos de prueba cargados correctamente.")
