# Estado del arte

Fichas de los artículos que revisamos para el proyecto de insatisfacción de clientes en Olist ([dataset en Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)).

La guía pide al menos 4 artículos, de revista si se puede (Elsevier o IEEE), y de cada uno: paradigma, técnicas, validación, métricas y resultados. En el informe cabe una página como máximo.

Cada ficha dice si leímos el artículo completo o solo el resumen. Con el resumen tenemos resultados, pero a veces nos falta la validación o los hiperparámetros.

## Los cuatro del informe

Los cuatro usan los datos de Olist. Los tres primeros predicen satisfacción; el cuarto predice abandono, y lo incluimos por el tema de fidelización.

### 1. Wong y Marikannan (2020). Memorias de congreso, IOP, acceso abierto

A.-N. Wong, B. P. Marikannan, "Optimising e-commerce customer satisfaction with machine learning," *Journal of Physics: Conference Series*, vol. 1712, art. 012044, 2020. DOI: [10.1088/1742-6596/1712/1/012044](https://doi.org/10.1088/1742-6596/1712/1/012044)

- Paradigma: supervisado, clasificación binaria (insatisfecho = 1 y 2; satisfecho = 3, 4 y 5).
- Técnicas: árbol de decisión, Random Forest, red neuronal y SVM, en R. Crean dos variables de entrega, `delivery_performance` (entrega real menos estimada) y `purchase_delivery_days` (días de compra a entrega). Comparan 20 variables contra las 5 más importantes y prueban cuatro técnicas de balanceo: submuestreo, sobremuestreo, SMOTE y ROSE.
- Validación: toman el 50 % del dataset para ahorrar cómputo y lo parten 70/30. A Random Forest le aplican validación cruzada de 1 a 5 pliegues.
- Métricas: accuracy, sensibilidad, especificidad, F1 y tiempo de cómputo.
- Resultados: todos los modelos quedan entre 87,0 % y 87,6 % de accuracy, con F1 de 0,93. La especificidad, que aquí mide cuántos insatisfechos detectan, va de 21,5 % a 29,7 %. El mejor es Random Forest (87,6 %, especificidad 29,5 %). Ninguna técnica de balanceo sube la especificidad, y eso nos llamó la atención. Las dos variables más importantes son las de entrega.
- Leído completo. Es de congreso, y la guía prefiere revistas.

### 2. Zaghloul, Barakat y Rezk (2024). Revista, Elsevier

M. Zaghloul, S. Barakat, A. Rezk, "Predicting E-commerce customer satisfaction: Traditional machine learning vs. deep learning approaches," *Journal of Retailing and Consumer Services*, vol. 79, art. 103865, 2024. DOI: [10.1016/j.jretconser.2024.103865](https://doi.org/10.1016/j.jretconser.2024.103865)

- Paradigma: supervisado, clasificación binaria con el mismo corte.
- Técnicas: regresión logística, SVM, Random Forest, Gradient Boosting y MLP.
- Datos: unen todas las tablas y quedan con 112 897 filas, una por artículo. Borran las filas con faltantes (2,6 %) y codifican las categóricas con label encoding.
- Validación: una sola partición aleatoria 67/33, sin validación cruzada. Eligen 20 variables con `SelectKBest` e información mutua, ajustan hiperparámetros con Grid Search y prueban RandomOverSampler, SMOTE y ADASYN.
- Métricas: AUC-ROC, AUC-PR, precisión, recall y F1 ponderados por clase, accuracy y tiempo de entrenamiento.
- Resultados: con la selección de variables, Gradient Boosting llega a 91 % de accuracy, F1 ponderado de 0,91 y AUC-ROC de 0,90. Con RandomOverSampler, Random Forest sube a 92 %, F1 ponderado de 0,92 y AUC-ROC de 0,90. Las variables más importantes son las de tiempo de entrega.
- Dónde no les creemos del todo: al trabajar por artículo, una orden con varios artículos puede quedar a la vez en entrenamiento y en prueba, y eso infla los resultados. Tampoco reportan métricas de la clase insatisfecha sola, y en el F1 ponderado pesa casi todo la clase mayoritaria. Nosotros trabajamos por orden.
- Leído completo. Es el más citado del grupo (68 citas).

### 3. Orman (2026). Revista, Elsevier, acceso abierto

R. Orman, "Hybrid deep learning for e-commerce customer satisfaction classification," *Array*, vol. 31, art. 101153, 2026. DOI: [10.1016/j.array.2026.101153](https://doi.org/10.1016/j.array.2026.101153)

- Paradigma: supervisado, clasificación binaria con el mismo corte.
- Datos: 108 449 filas "a nivel de transacción", con 91 754 satisfechos y 16 695 insatisfechos (15,4 %). Son más filas que órdenes en la base (unas 99 000), así que por más que el autor diga que consolidó artículos y pagos, alguna orden sigue repetida. Deja fuera el texto de las reseñas para no filtrar la etiqueta.
- Variables: 11 grupos que al codificarse dan 41 columnas. Incluyen el estado de la orden, ganancia bruta y margen (calculados como residuo del pago), valor total, precio, flete, volumen, estado y ciudad del vendedor, y ciudad del cliente. Ninguna variable de tiempo de entrega. Codifica estado de la orden y del vendedor con one-hot, las ciudades por frecuencia (calculada solo en entrenamiento) y escala las numéricas con min-max.
- Técnicas: 8 modelos clásicos (árbol de decisión, regresión logística, Naive Bayes, Random Forest, CART, CatBoost, XGBoost, LightGBM) contra 5 redes (CNN, RNN, MLP, GRU-LSTM, LSTM-GRU).
- Validación: partición estratificada 80/20 (`random_state = 42`), con 86 759 filas de entrenamiento y 21 690 de prueba. Los modelos clásicos se eligen con validación cruzada estratificada de 3 pliegues; las redes, con una partición de validación aparte. SMOTE solo sobre el entrenamiento ya procesado. Reporta intervalos de Wilson al 95 % para la accuracy de prueba.
- Hiperparámetros: XGBoost, LightGBM y CatBoost con 300 árboles y tasa de aprendizaje de 0,05; Random Forest y CART con profundidad 5; árbol de decisión con profundidad 15. GRU-LSTM: capa GRU de 50 unidades, capa LSTM de 50 y salida sigmoide, con Adam, entropía cruzada binaria, 10 épocas y lotes de 32.
- Métricas: accuracy, precisión, recall y F1 por clase, macro-F1, F1 ponderado, balanced accuracy, MCC, AUC-ROC y AUC-PR.
- Resultados: gana GRU-LSTM con 86,12 % de accuracy (IC 95 %: 85,65-86,57) y F1 de 92,36 % en la clase satisfecha. De 3339 insatisfechos en prueba detecta 479: recall de 14,35 %, precisión de 76,03 %, F1 de 24,14 %. Macro-F1 de 58,25 %, balanced accuracy de 56,76 %, MCC de 0,29. El AUC-ROC es 0,5625, casi el de una moneda al aire, y el AUC-PR de la clase insatisfecha es 0,23 frente a una prevalencia de 0,15. Entre los clásicos gana Naive Bayes (85,77 %); Random Forest queda en 80,40 %.
- Importancia de variables (Random Forest): el estado de la orden pesa 0,46, seguido de ganancia bruta (0,16) y margen (0,15). O sea, el modelo aprende sobre todo que las órdenes canceladas o no entregadas salen mal calificadas.
- Nuestra lectura: es la referencia más honesta del grupo, porque reporta todo por clase y cuida la fuga de información. Su AUC de 0,56 muestra lo poco que se logra sin variables de entrega; en nuestro análisis exploratorio, `dias_entrega` sola ya tiene un AUC univariado de 0,685. También confirma que un accuracy de 86 % no dice nada si el modelo casi no encuentra insatisfechos.
- Leído completo.

### 4. Zeinali, Ramezani Asli y Khalili (2026). Revista, Wiley, acceso abierto

M. Zeinali, L. Ramezani Asli, M. A. Khalili, "Integrating Business Intelligence and CRM Systems With a Machine Learning Approach for Predictive Customer Retention in E-Commerce," *The Scientific World Journal*, 2026. DOI: [10.1155/tswj/1946904](https://doi.org/10.1155/tswj/1946904)

- Paradigma: supervisado, clasificación binaria por cliente (abandono sí o no). Antes, segmentan clientes con K-means sobre recencia, frecuencia y monto (RFM).
- Etiqueta: abandono si el cliente no compra en 180 días. Calculan las variables solo con datos anteriores a la fecha de corte (la última fecha menos 180 días).
- Técnicas: regresión logística, Random Forest y XGBoost, con pesos por clase (`class_weight`, `scale_pos_weight`).
- Validación: partición estratificada 80/20 por cliente, y validación cruzada estratificada de 5 pliegues para los hiperparámetros.
- Métricas: accuracy, precisión, recall, F1 y AUC-ROC.
- Resultados: XGBoost logra accuracy de 0,81, precisión de 0,79, recall de 0,83, F1 de 0,81 y AUC de 0,85. Random Forest queda en 0,76, F1 de 0,72 y AUC de 0,76. La calificación promedio del cliente es la segunda variable más importante, después de la recencia.
- Algo no nos cuadra: si el 97 % de los clientes compra una sola vez, no entendemos cómo llegan a un `scale_pos_weight` de 1,8. Hay que revisarlo antes de apoyarnos mucho en sus números.
- Leído completo (versión en PMC). Lo usamos para sustentar el vínculo entre satisfacción y fidelización, que en nuestros datos no pudimos medir.

## Complementarios

### Wangkiat y Polprasert (2023). Congreso, IEEE

P. Wangkiat, C. Polprasert, "Machine Learning Approach to Predict E-commerce Customer Satisfaction Score," en *Int. Conf. on Business and Industrial Research (ICBIR)*, IEEE, 2023. DOI: [10.1109/ICBIR57571.2023.10147542](https://doi.org/10.1109/ICBIR57571.2023.10147542)

- Usa Olist con 4 clases (Low, Average, Good, Excellent), casi todas las órdenes en Excellent.
- Random Forest, regresión logística y KNN contra un modelo base que predice con la calificación promedio del producto. Sus variables principales son la duración de la entrega y la calificación promedio del producto en otras compras.
- El mejor, Random Forest, llega a precisión de 0,34, recall de 0,36 y macro-F1 de 0,32. Las variables más importantes son la media (0,313) y la desviación estándar (0,087) de la calificación del producto.
- Solo leímos el resumen: el PDF requiere acceso institucional de IEEE y no sabemos la validación. En el informe lo citamos para justificar por qué no usamos 5 clases, ya que con 4 obtienen un macro-F1 de 0,32 contra 0,58 de Orman en el caso binario.

### Ravula (2023). Revista, Springer

P. Ravula, "Impact of delivery performance on online review ratings: the role of temporal distance of ratings," *Journal of Marketing Analytics*, vol. 11, no. 2, pp. 149-159, 2023. DOI: [10.1057/s41270-022-00168-5](https://doi.org/10.1057/s41270-022-00168-5)

- No usa aprendizaje de máquina: estima un logit ordinal bayesiano sobre 942 órdenes emparejadas por puntaje de propensión.
- Dice que los datos vienen de "una empresa de un mercado emergente" y nombra a Olist solo como ejemplo de plataforma, así que no podemos afirmar que use nuestro dataset.
- Un día de retraso baja la calificación (coeficiente de -0,610) unas 2,5 veces más de lo que la sube un día de adelanto (+0,245). Calificación promedio: 4,41 si llega antes, 4,16 a tiempo, 3,45 tarde.
- Lo podemos citar en la sección 2 como respaldo de que el retraso pesa en la calificación.
- Leído completo.

### Descartado por ahora

Asfe, Rahman y Hossain (2025), "MNeuralTab...", *Discover Applied Sciences*, DOI [10.1007/s42452-025-07157-0](https://doi.org/10.1007/s42452-025-07157-0). Predicen abandono en Olist con 99,62 % de accuracy y AUC de 0,98. Ese resultado nos parece demasiado alto para esta base; sospechamos fuga de información en la forma de definir el abandono. Si lo citamos, hay que leerlo con lupa.

## Lo que sacamos de la revisión

1. Los tres trabajos de satisfacción binaria (Wong, Zaghloul, Orman) usan el mismo corte: 1 y 2 contra 3, 4 y 5. Wangkiat usa 4 clases y le va mucho peor.
2. Wong y Zaghloul ponen las variables de entrega como las más importantes. Zeinali, que predice abandono, pone primero la recencia y el retraso de la entrega en cuarto lugar.
3. El desbalance es lo que más pesa en la evaluación. Wong y Orman reportan accuracy de 86 % a 88 %, pero detectan entre 14 % y 30 % de los insatisfechos, y el mejor modelo de Orman tiene un AUC-ROC de apenas 0,56. Si solo reportamos accuracy nos engañamos, así que nuestras métricas principales serán el recall y el F1 de la clase insatisfecha, la balanced accuracy y el AUC.
4. En Wong y Zaghloul ganan los ensambles de árboles (Random Forest, Gradient Boosting). En Orman gana una red GRU-LSTM, pero con 14 % de recall en la clase que nos interesa y sin variables de entrega.
5. Las variables de entrega marcan la diferencia. Los trabajos que las usan (Wong, Zaghloul) las encuentran como las más importantes, y el que no las usa (Orman) queda cerca del azar en AUC.

## Métricas que no se vieron en clase

La clase 5 del curso define sensibilidad, precisión, especificidad, exactitud, $F_\beta$, MCC, exactitud balanceada, G-mean y la curva ROC. Las que faltan ya están definidas en `informe/informe.tex`:

- F1 macro: $\frac{1}{K}\sum_{k=1}^{K} F1_k$
- F1, precisión y sensibilidad ponderadas: $\sum_k \frac{n_k}{N} M_k$
- AUC-ROC: $\int_0^1 TPR \, d(FPR)$ (la clase lo menciona sin definirlo)
- AUC-PR: $\int_0^1 P \, dR$

## Pendientes

- [x] Leer el PDF de Zaghloul et al. (2024).
- [x] Revisar si Ravula (2023) usa Olist: no lo dice.
- [x] Elegir los cuatro del informe: Wong, Zaghloul, Orman y Zeinali.
- [x] Leer el PDF de Orman (2026).
- [ ] Buscar en Google Scholar los artículos que citan el dataset, por si aparecen más revistas de Elsevier o IEEE.
