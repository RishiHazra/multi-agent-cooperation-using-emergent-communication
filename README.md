# multi-agent cooperation using emergent communication: 

* multi-agent reinforcment learning
* emergent-social-structures

## Description

Project describing how social complexity relates to communication complexity. Each agent gets a cropped portion of an image and have to cooperate with other agents to reach a prediction. A group of agents combine to form a community. Based on the input they recieve, they can either be a image-community or a text community. Agents are also characterized by certain traits

## Datset used
* CIFAR-10

## Results
Using a Parzen window estimation, I have been able to show the following for different group structures:  **Social complexity directly influence communication complexity**

<p align="center">
  <img src="https://user-images.githubusercontent.com/36856187/117822370-cbc79980-b26c-11eb-8f33-273d9b04d2f3.png"  width="200" alt="1"/>
  <img src="https://user-images.githubusercontent.com/36856187/117822785-1fd27e00-b26d-11eb-978b-b2195e90c266.png"  width="200" alt="2"/>
  <img src="https://user-images.githubusercontent.com/36856187/117822858-2e209a00-b26d-11eb-8d93-8f71fd1629be.png"  width="200" alt="3"/>
  <img src="https://user-images.githubusercontent.com/36856187/117822936-3e387980-b26d-11eb-8ac4-d41ca04d86b0.png"  width="200" alt="4"/>
</p>

* Two groups with an identical number of social roles and similar relationship structure may differ from one another in the number of individuals, with larger groups being more complex than smaller groups. 

* Two groups with an identical number of individuals and similar relationship structure may differ from one another in the number of social roles, with groups having more social roles being more complex than groups with fewer social roles. 

* Finally, two groups with an identical number of individuals and identical number of social roles may differ from one another in their relationship structure, with groups having more dyadic, triadic and higher level relations among individuals being more complex than groups having fewer such relations. 
