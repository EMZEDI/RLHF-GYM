# RLHF-GYM
RLHF Wrapper for OpenAI's Gym Library

# First time run

```bash
git clone <REPO>
cd RLHF-GYM
chmod u+x build.sh run.sh
./build.sh
```

- For the second run so on, you can just run the following commmand:
    `./run.sh`

- The `app` directory of the container is mounted to the `app` directory of the host machine. So, you can edit the code in the host machine and run it in the container.

- Please add any new library that you work with to the requirements.txt file.