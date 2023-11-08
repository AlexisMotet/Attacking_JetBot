# Attacking JetBot (2022 - 2023)

The goal of this engineering school project was to attack a JetBot embedded image classifier (https://jetbot.org/master/examples/collision_avoidance.html) with an adversarial patch.

The embedded model has an AlexNet architecture. It was trained by two other students in a previous project. It is capable of classifying images containing an obstacle. When coupled with a guidance algorithm, it enables the robot to move autonomously without running into obstacles. Our goal was to create a patch to be applied to an obstacle so that it would not be recognized by the model and the robot would crash. We worked with access to the model and the training data. 

The patch is obtained through stochastic gradient descent. At each iteration, we insert the patch into a training image, then calculate the gradient of an attack score with respect to the attacked image through model backpropagation. Next, we clip the computed gradient over the entire image to modify only the patch. By applying various linear and non-linear transformations, such as distortion, we have created a robust patch capable of working across different scenes.

In the demonstration video, the robot has a guidance algorithm that makes it stop when the model detects an object. When a white or random patch is used, the robot stops before the wall. When the adversarial patch is used, the robot goes straight into the wall.

https://user-images.githubusercontent.com/84445302/222529376-25040305-69b3-438d-8827-54c12ebd1182.mp4

The full video and two others can be found here : https://drive.google.com/drive/folders/14Z4wmOgZl15x2TCZS4fOQZop32LfqaFO?usp=share_link. 

More information can be found in our [report](RapportEcrit_SouhailaNouinou_AlexisMotet.pdf).



