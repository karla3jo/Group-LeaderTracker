# Traking Group-Leader social interactions
Released code for the paper "Towards robots reasoning about group behavior of museum visitors: Leader detection and group tracking".
https://content.iospress.com/articles/journal-of-ambient-intelligence-and-smart-environments/ais467

This code is open-source and free for anyone to experiment further with it :-) 
If you publish any work that uses this software or refers to it in any way, please just cite us:

Trejo, K., Angulo, C., Satoh, S. I., & Bono, M. (2018). Towards robots reasoning about group behavior of museum visitors: Leader detection and group tracking. Journal of Ambient Intelligence and Smart Environments, 10(1), 3-19.

## Requisites
- OpenCV 2.4.11
- Dlib 18.18
- Visual Studio (C# environment)
- An input video sequence (.avi file)

## How to use
Just run the .exe file at *x64/Release* directory from any version (group_leader_traker or exponential_group_leader_traker) in command line as: **chosen_version.exe path_to_your_video_file.avi**

If you want to edit any of the source code, open the desired version of the project in Visual Studio and add the corresponding .prop files and dependencies --for DLib and OpenCV-- to your project path by importing the contents of *libraries* directory.

In case you are willing to generate ground truth to validate your experiments, just remember to uncomment the corresponding section in the main files of group_leader_traker/exponential_group_leader_traker.

We cannot provide the image-video datasets from our paper. However, you can still watch the algorithm in action for the Miraikan museum video sequences at https://www.youtube.com/watch?v=gb8dIY-UiTg&t=2s

## Contact
Please leave a comment in my YouTube channel or ResearchGate page* if you have any questions or require assistance of any kind. I'll be glad to help!

*https://www.researchgate.net/publication/322609811_Towards_robots_reasoning_about_group_behavior_of_museum_visitors_Leader_detection_and_group_tracking/comments
