# foodalike

have u ever been called a lobster bisque? do boys dm u about ur baby back ribs? find out which scrumptious food u r!!!

[check out our devpost](https://devpost.com/software/f-o-o-d-a-l-i-k-e)   
[check out our app on heroku](https://foodalike4.herokuapp.com/)
[our android frontend](https://github.com/marievu/Foodalike4)   

### what it is

Take a selfie from the foodalike app and find out what food you look like ;)

We chose 24 categories from the [food-101 dataset](http://www.vision.ee.ethz.ch/datasets_extra/food-101/) and trained a Convolutional Neural Network with two laters over 10 epochs. We achieved an accuracy of ~ 65% and a validation accuracy of ~ 35%. We then ~~set up a python backend~~  tried to set up a python server on heroku with flask to handle http post calls to get predictions from the trained model. Our android app sends the user's selfie as an http requests to the backend and gets the category name as a response. It then outputs a short description

### who we are

[Lara Kollokian](https://github.com/larakollokian)    
[Marie Vu](https://github.com/marievu)    
[Luke Ma](https://github.com/lukewma)   
[Ethan Itovitch](https://github.com/cheeth_keeth)   

### what's next

We made an improved model and we want to train it on a more powerful computer to get better predictions. We also want to successfully connect the server to the android app. Our backend currently only runs locally :(
