# CSE3000-REVISE-dynamics-quantifier
This is the code used in the Research Project to quantify the endogenous shifts in a model and domain when using the REVISE recourse generator.

# Usage
1. clone the [CARLA - Counterfactual And Recourse Library](https://github.com/carla-recourse/CARLA) from it's git repo inside this project root folder:

    ```
    git clone https://github.com/carla-recourse/CARLA.git
    ```

2. Build the Docker image defined in the `Dockerfile`:

    ```
    docker build --tag carla:latest
    ```

3. Run the Docker container:

    ```
    docker run -v ${PWD}:/code --rm carla
    ```

4. Read the output written in the newly created csv file inside the project's root folder.
