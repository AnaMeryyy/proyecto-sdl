import cv2
import numpy as np

# ============================================================
# PLANTILLA DE PROYECTO SDL
# ============================================================
# API disponible en sdl:
#
# INFERENCIA (puedes llamar a varios en el mismo process()):
#   sdl.preload("backend")              ← precarga en prepare()
#   results = sdl.infer("backend", frame)
#
# BACKENDS DISPONIBLES:
#   "yolo_detect"  → {n_detections, detections:[{class,conf,box}]}
#   "tracking"     → {tracker_ids:[int], boxes:[[x1,y1,x2,y2]], class_ids:[int], confidences:[float], names:[str]}
#   "depth"        → {depth_map (HxW float32)}
#   "classify"     → {top1, top1_conf, top5:[{class,conf}]}
#   "face"         → {n_faces, faces:[{box,name,conf,embedding}]}
#   "clip"         → {scores:{label:float}}
#   "bgsubtract"   → {mask (HxW uint8), n_contours, contours}
#   "flow"         → {flow (HxW×2 float32), magnitude, angle}
#   "hands_yolo"   → {hands:[{handedness,box,landmarks:[{px,py,x,y,z}×21],lm_names}]}
#   "hands_mp"     → {hands:[{handedness,box,score,landmarks:[{px,py,x,y,z}×21],lm_names}]}  (MediaPipe ONNX)
#   "face_mesh"    → {n_faces, faces:[{box,conf,score,landmarks:[{px,py,z}×468]}]}
#   "license_plate"→ {n_plates, plates:[{box,conf,text,ocr_conf}]}
#   "emotion"      → {n_faces, faces:[{box,emotion,scores:{happy,...}}]}
#   "yolo_world"   → {n_detections, classes, detections:[{class,conf,box}]}
#   "aruco"        → {n_markers, markers:[{id,center,corners}]}
#   "age_gender"   → {n_faces, faces:[{box,gender,gender_conf,age_range}]}
#
# PERSISTENCIA (persiste entre sesiones en disco):
#   sdl.save("clave", valor)            ← guarda cualquier valor JSON
#   valor = sdl.load("clave", default)  ← carga valor guardado
#   sdl.data_dir                        ← ruta al directorio de datos
#
# TIEMPO Y FRAMES:
#   sdl.elapsed()                       ← segundos desde carga del proyecto
#   sdl.frame_count                     ← número de frames procesados
#
# LOGGING:
#   sdl.log("mensaje")                  ← visible en el panel de estado
#
# HELPERS DE DIBUJO:
#   sdl.draw_box(img, box, label, color)        ← bounding box
#   sdl.draw_landmarks(img, landmarks, color)   ← puntos de mano
#   sdl.draw_text(img, texto, pos, color)       ← texto libre
#
# WIDGETS (panel de control en la interfaz):
#   Widgets disponibles:
#     toggle1, toggle2          → bool  (interruptor ON/OFF)
#     counter1, counter2        → int   (número configurable)
#     slider1                   → float 0.0-1.0 (deslizador)
#     led1, led2, led3          → bool  (indicador visual)
#     text1, text2              → str   (texto de salida)
#     text_input1, text_input2  → str   (texto de entrada del alumno)
#     file1, file2              → visor de fichero (descarga + vaciado)
#     select1, select2          → desplegable (opciones configurables)
#
#   Métodos comunes:
#     .label("etiqueta")        ← cambia la etiqueta visible en el panel
#     .value                    ← lee el valor actual del widget
#     .value(x)                 ← escribe el valor (text, led, counter, toggle)
#     .hide()                   ← oculta el widget en el panel
#     .show()                   ← muestra el widget en el panel
#     .file("/ruta/fichero")    ← asocia fichero al widget file_view
#     .options(["a","b","c"])   ← define opciones del desplegable select
#
#   Ejemplos:
#     activo = sdl.widget("toggle1").value           # bool
#     umbral = int(sdl.widget("counter1").value)     # int
#     nivel  = float(sdl.widget("slider1").value)    # float 0-1
#     texto  = str(sdl.widget("text_input1").value)  # str
#     sdl.widget("text1").value(f"n={n}")            # output
#     sdl.widget("led1").value(n > umbral)           # LED
#     sdl.widget("toggle2").hide()                   # ocultar
#     sdl.widget("file1").file(sdl.data_dir+"/log.csv")  # visor
#     sdl.widget("select1").options(["op1","op2","op3"])  # desplegable
#     opcion = str(sdl.widget("select1").value)           # leer selección
# ============================================================

def prepare(sdl):
    """Se ejecuta UNA VEZ al cargar el proyecto. Configura widgets y precarga modelos."""
    # Configurar etiquetas y valores iniciales de los widgets que uses
    sdl.widget("counter1").label("Penalizacion movil").value(10).show()
    #el valor depende de las personas que hayan, si hay muchas el valor tiene que ser más bajo
    sdl.widget("slider1").label("Umbral baja atencion").value(50).show()
    sdl.widget("led1").label("Baja atencion").show()
    sdl.widget("text1").label("Indice atencion").show()
    sdl.widget("text2").label("Estado").show()
    
    sdl.widget("toggle1").label("Activar")
    sdl.widget("counter1").label("Umbral").value(10)
    sdl.widget("slider1").label("Sensibilidad")
    sdl.widget("led1").label("Alerta")
    sdl.widget("text1").label("Estado")
    sdl.widget("text_input1").label("Buscar objeto")
    # Ocultar los widgets que no uses en este proyecto
    sdl.widget("toggle2").hide()
    sdl.widget("counter2").hide()
    sdl.widget("led2").hide()
    sdl.widget("led3").hide()
    sdl.widget("text2").hide()
    sdl.widget("text_input2").hide()
    sdl.widget("file1").hide()
    sdl.widget("file2").hide()
    sdl.widget("select1").hide()
    sdl.widget("select2").hide()
    # Para volver a mostrar: sdl.widget("toggle2").show()
    # Asociar un fichero al widget file_view
    # sdl.widget("file1").label("Mi log").file(sdl.data_dir + "/datos.csv")
    # Configurar un desplegable
    # sdl.widget("select1").label("Modo").options(["deteccion", "seguimiento", "clasificacion"])

    # Precargar modelos (descomenta los que uses)
    sdl.preload("yolo_detect")
    sdl.preload("tracking") #podríamos usarlo para ids persistentes 
    # sdl.preload("depth")
    # sdl.preload("classify")
    sdl.preload("face")
    # sdl.preload("clip")
    # sdl.preload("bgsubtract")
    # sdl.preload("flow")
    sdl.preload("hands_yolo") #se podría usar para que no solo detecte móviles en la mesa sino manos usándolos
    # sdl.preload("hands_mp")
    # sdl.preload("face_mesh") detecta no solo la cara sino los puntos se podría utilizar para analizar con más detalle
    # sdl.preload("license_plate")
    # sdl.preload("emotion")
    # sdl.preload("yolo_world")
    # sdl.preload("aruco")
    # sdl.preload("age_gender")

    # Cargar datos persistidos de sesiones anteriores
    # mi_contador = sdl.load("contador", default=0)

def process(frame, sdl):
    """Se ejecuta en cada frame. Retorna imagen BGR anotada para AI Output."""
    img = frame.copy()

    # Leer widgets
    activo = sdl.widget("toggle1").value
    umbral_baja = float(sdl.widget("counter1").value)
    # texto_buscado = str(sdl.widget("text_input1").value)  # entrada de texto del alumno

    if not activo:
        sdl.draw_text(img, "Sistema pausado", (10, 30), color=(100,100,100))
        sdl.widget("text1").value("Pausado")
        sdl.widget("text2").value("Sistema inactivo")
        sdl.widget("led1").value(False)
        return img

    # --- Ejemplo: detección de objetos ---
    # results = sdl.infer("yolo_detect", frame)
    # for det in results.get("detections", []):
    #     if det["conf"] >= umbral / 100.0:
    #         sdl.draw_box(img, det["box"], f"{det['class']} {det['conf']:.2f}")

    # --- Ejemplo: múltiples backends en el mismo frame ---
    # r1 = sdl.infer("yolo_detect", frame)
    # r2 = sdl.infer("emotion", frame)  # ← se puede llamar a varios backends

    # --- Ejemplo: persistencia entre sesiones ---
    # contador = sdl.load("contador", default=0)
    # contador += 1
    # sdl.save("contador", contador)
    # sdl.widget("text1").set(f"Total: {contador}")

    # --- Ejemplo: temporizador ---
    # sdl.draw_text(img, f"Tiempo: {sdl.elapsed():.1f}s  Frame: {sdl.frame_count}", (10, 30))

    # --- Tu código aquí ---

    # Inferencias
    res_tracking = sdl.infer("tracking", frame)
    res_faces = sdl.infer("face", frame)
    res_objetos = sdl.infer("yolo_detect", frame)
    res_hands = sdl.infer("hands_yolo", frame)

    #obtengo las personas totales, sus caras y los objetos de la grabación
    personas = obtener_personas_tracking(res_tracking)
    caras= res_faces.get("faces", [])
    objetos = res_objetos.get("detections", [])
    
    #de esos objetos hay que identificar cuales que son móviles
    moviles = [
        d for d in objetos
        if d["class"] in ["cell phone", "phone", "mobile phone"]
    ]

    #manos detectadas
    manos = res_hands.get("hands", [])

    #contamos las personas y las caras
    n_personas = len(personas)
    n_caras = len(caras)

    #que gente está mirando a otro lado
    personas_sin_cara= max(0, n_personas - n_caras)

    #que móviles están cerca de una mano llamo al método contar_moviles_en_mano
    moviles_en_mano = contar_moviles_en_mano(moviles, manos)

    #para las distracciones graves queremos hacer personas con móvil cerca en la mano y que no este mirando hacia delante
    distracciones_graves = min()
    

    


    return img
