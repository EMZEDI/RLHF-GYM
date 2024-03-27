# the following will mount the working directory to the app dir of the container
docker run -it -p 4000:80 -v $(pwd):/app rlhf-gym bash