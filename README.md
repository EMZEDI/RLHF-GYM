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

# Project Structure

- `app` directory contains the source code of the project in the container. The following are in this directory:

    - `dataset` directory includes the synthetic dataset that is used in the project to mimic the human preference data for the reward modeling.

    - `models` directory includes the trained models that are used in the project.

    - `src` contains the source code for the RLHF specific wrappers and the training code for the RLHF.

        - `data/synthetic-generation.py` is the code to generate the synthetic preference dataset.

        - `environment.py` is the MiniGrid environment wrapper for the RLHF.

        - The rest of the scripts belong to the RLHF training code.