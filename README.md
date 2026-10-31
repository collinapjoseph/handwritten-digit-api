# Handwritten Digit API
Collin Joseph <br>
## Run
### Docker
1. Pull Docker image from Docker Hub [collinapjoseph/handwritten-digit-api](https://hub.docker.com/repository/docker/collinapjoseph/handwritten-digit-api/general)
2. Build image to a container
3. Run container: `docker run -p 8000:8000 collinapjoseph/handwritten-digit-api`
### Local
1. Clone this repository.
2. Install dependencies: `pip -r requirements.txt`
3. Run mnist_cnn.py to train and save ONNX formated model.
4. Run REST API server: `uvicorn server:app --port 8000`
## Use
### UI
1. Navigate to http://localhost:8000
2. Use mouse/pointer to draw digits.
3. Click "classify" to upload drawing and receive classification.
### API Endpoint
Upload images of digits to /cnn route using POST requests. <br>
The server expects the image to be included as `form-data` in the `body` of the request with the key: `file`. <br>
Example, Curl command: `curl -X POST http://localhost:8000/cnn -F "file=@digit.png"`<br>
Example, Postman instructions: <br>
1. Create a post request to http://localhost:8000/cnn
2. In the body of the request, add a form-data field of type `file`
3. Set the key of the field to "file" and attach the image you wish to upload.
4. Send the request and wait for the response.
