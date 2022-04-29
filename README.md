# omrtorch
run server :

 ```
FLASK_ENV=development FLASK_APP=app.py flask run
 ```

from another terminal tab give input: 
 ```
curl -X POST -F "file=@/Users/abdullahkucuk/input_pic.jpg" http://localhost:5000/predict
 ```
