
## Notas:
Analiza el proyecto y las librerias / funciones
que aqui se utilizan y puedas instalarlas para que pruebes el sistema.

## Curso Seminario de tesis 2

Necesito de que puedas realizar un pequeño análisis del estado del arte
tesis, proyectos, papers presentados que abarquen este problema de
la misma forma o similar.
Luego necesito de que me entregues un notebook en donde puedas validar
mediante machine learning causal alguno de los puntos que se abarcan 
en el notebook1 o el que creas que pueda aportar más valor. Además
de que puedas realizar algunos cuantos grafos de dependencia de lo que vas a evaluar.


// Consideraciones relevantes 

1. Sí: "todo contra todos" es básicamente lo que hace Double ML

No tienes que ir variable por variable. Metes todas las variables observadas relevantes (área, distrito, coordenadas, antigüedad...) y el modelo ajusta por ellas a la vez. Una regresión múltiple ya hace eso.

Lo que añade Double ML es que usa ML flexible en vez de una línea recta, para capturar relaciones no lineales sin que tú las especifiques.

2. Sobre los residuos: no es "la influencia de esa variable"

Lo que describes es casi exacto, pero conviene afinar. Los residuos se calculan así:

Residuo de baños = la parte de los baños que no se explica por área, ubicación, etc.
Residuo de precio = la parte del precio que no se explica por área, ubicación, etc.

Luego se mira si ambos residuos se mueven juntos. El residuo no es "la influencia de los baños". Es la variación que sobra una vez que quitaste todo lo explicado por los controles.

Para llamar causal a esa relación hay que suponer que esa variación sobrante es como si fuera aleatoria: que dos departamentos iguales en todo lo observado tienen distinto número de baños "por azar" y no por algo oculto. Eso es una presunción, no un resultado. Tienes razón en eso.

3. Por qué "todo versus todos" no basta

Hay dos razones distintas.

Razón A: más controles no siempre es mejor. Hay variables que no debes incluir. Tres tipos:

Tipo	Relación	¿Controlar?
Confusor	causa a T y a Y (T = tratamiento, Y = resultado), por ejemplo el área	Sí
Mediador	está entre T y Y: T → M → Y	No, porque bloqueas parte del efecto que quieres medir
Colisionador	es consecuencia de T y de Y: T → C ← Y	No, porque crea una relación falsa donde no la había

Un ejemplo con tus datos: precioxm2 se calcula a partir del precio. Si lo metes como control para estimar el efecto de un baño sobre el precio, estás controlando por una consecuencia del resultado y distorsionas todo. Es un caso extremo, pero la lógica es la misma con variables menos obvias.

Razón B (la de fondo): los datos solos no te dicen cuál es cuál. Mira estas tres historias:

 A → B          A ← B          A ← C → B

Las tres producen la misma correlación entre A y B. Con solo los datos no hay forma de saber cuál es la verdadera. Para decidir si una variable es confusor, mediador o colisionador necesitas saber qué causa a qué, y eso viene de conocimiento externo, no de los números.

La regla práctica que usa casi todo el mundo es el orden temporal: controla por lo que ocurre antes del tratamiento y afecta a ambos, y no controles por lo que ocurre después. Por ejemplo, la ubicación y la antigüedad existen antes de decidir cuántos baños tiene el departamento.

4. Lo que de verdad no se puede probar

Aunque incluyas todas las variables que tienes, queda un problema que ninguna prueba resuelve. Faltan variables que no están en la base: acabados, piso, vista, seguridad. Si alguna de ellas afecta tanto a los baños como al precio, el supuesto de "variación como aleatoria" se rompe, y los datos no te avisan.

Eso se llama confusor no observado. Por eso el PDF insiste en la fase de robustez: no puedes demostrar que no existe, pero puedes medir cuán fuerte tendría que ser para tumbar tu resultado (E-value, refutadores).

5. ¿Y los algoritmos que prueban todo contra todo?

Existen. Es la propuesta 5 del PDF (PC, LiNGAM, NOTEARS), que hace pruebas de independencia entre pares y grupos de variables para aprender el DAG desde los datos. Tienen dos límites:

Asumen que no hay confusores ocultos, que es justo lo que no puedes garantizar.
Muchas veces no logran orientar las flechas. Dicen "A y B están conectadas" pero no "A → B".

Por eso el PDF los pone como chequeo de tu DAG y no como sustituto.

En una frase

No es que no se pueda probar todo contra todos. Es que, aunque lo hagas, los datos no te dicen qué variables deben entrar ni si falta alguna. Eso es un supuesto causal que tú argumentas, y el trabajo serio consiste en defenderlo y ponerlo a prueba.

Ese es el precio de hacer causalidad con datos observacionales: no hay forma de evitar los supuestos, solo de hacerlos explícitos y comprobar qué tan frágil es tu resultado frente a ellos. En una tesis, mostrar eso es parte del valor del trabajo.

Si quieres, tomamos tus variables reales una por una (area, dormitorios, antiguedad, distrito, type, precision...) y las clasificamos como confusor, mediador o dudosa para el efecto de los baños. Es un buen ejercicio para ver dónde están las decisiones difíciles.