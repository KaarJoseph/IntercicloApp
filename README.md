# FilterApp

## Universidad Politécnica Salesiana  
**Carrera de Computación**  
**Proyecto Interciclo**

---

## Descripción
**FilterApp** es una aplicación móvil desarrollada como parte del proyecto interciclo en la **Universidad Politécnica Salesiana**. Esta herramienta permite a los usuarios procesar imágenes aplicando filtros de convolución acelerados por GPU mediante **PyCUDA**, así como publicar, dar "like" y comentar publicaciones en un feed dinámico. La aplicación está diseñada para explorar conceptos avanzados de computación paralela, desarrollo móvil y dockerización.

---

## Características Principales
- **Procesamiento de Imágenes**: Aplicación de filtros de convolución personalizados, incluyendo filtros únicos con el logo de la UPS.
- **Publicaciones Sociales**: Los usuarios pueden subir imágenes al feed, comentarlas y dar "like".
- **Feed Dinámico**: Visualización de publicaciones ordenadas por fecha, con actualización en tiempo real.
- **Integración con PyCUDA**: Uso de GPU para acelerar el procesamiento de imágenes.
- **API Dockerizada**: Gestión eficiente del backend mediante contenedores Docker.
- **Interfaz Amigable**: Diseño intuitivo que mejora la experiencia del usuario.

---

## Tecnologías Utilizadas
- **Frontend (App móvil)**:
  - Android (Kotlin).
  - XML para diseño de interfaz.
- **Backend (API)**:
  - Flask (Python).
  - PyCUDA para procesamiento de imágenes.
  - Docker para la contenedorización.
- **Base de Datos**:
  - PostgreSQL para almacenamiento de usuarios y publicaciones.

---

## Requisitos Previos
1. **Entorno de Desarrollo**:
   - Android Studio.
   - Python 3.12.7.
2. **Dependencias**:
   - Flask
   - PyCUDA
   - NumPy
   - PostgreSQL
3. **Docker**:
   - Docker instalado y configurado para la API y base de datos.

---

## Instrucciones de Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/KaarJoseph/filterapp.git
cd filterapp
