#include <Servo.h>

Servo myservo1;
Servo myservo2;

int pos1 = 0;
int pos2 = 0;

const int dirPin  = 5;
const int stepPin = 4; 
const int enPin   = 3;

const int pinX = A1; // Joystick X axis
const int pinY = A0; // Joystick Y axis
const int sw = A3; // Joystick switch
void setup() {
  pinMode(pinX, INPUT);
  pinMode(pinY, INPUT); 

  myservo1.attach(10);
  myservo2.attach(9);

  pinMode(stepPin,OUTPUT); 
  pinMode(dirPin,OUTPUT);
  pinMode(enPin,OUTPUT);
  Serial.begin(9600);
  
  digitalWrite(enPin,LOW);
}
 
void loop() {
  int X = analogRead(pinX);
  int Y = analogRead(pinY);
  int sw = digitalRead(A3);
  Serial.print(X);
  Serial.print("\t"); 
  Serial.println(Y);

  if(Y < 200){ 
    stepper_mov_1();
  }

  if(Y > 900){
    stepper_mov_2();
  }

    if(X < 450){
    servo_mov_1();
  }

  if(X > 570){
    servo_mov_2();
  }

  if(Y < 450 && Y > 200){
    servo_claw_2();
    }
  
  if(Y > 570 && Y <900){
    servo_claw_1();
    }

}

//ANTI-CLOCKWISE MOVEMENT of ELBOW
void stepper_mov_1(){
  digitalWrite(dirPin,LOW);

  while(1){
    int Y = analogRead(pinY);
     
    digitalWrite(stepPin,LOW); 
    delayMicroseconds(1000); 
    digitalWrite(stepPin,HIGH); 
    delayMicroseconds(1000); 

    if(Y > 200){
      break;
      }
    }
  }

//CLOCK-WISE MOVEMENT of ELBOW
void stepper_mov_2(){
    digitalWrite(dirPin,HIGH);
   
    while(1) {
    int Y = analogRead(pinY);
    
    digitalWrite(stepPin,LOW); 
    delayMicroseconds(1000); 
    digitalWrite(stepPin,HIGH); 
    delayMicroseconds(1000); 

    if(Y < 900){
      break;
      }
    }
  }

// CLOCK-WISE MOVEMENT OF SERVO
void servo_mov_1(){
  while (pos1 <= 180) {                 // goes from 0 degrees to 180 degrees
    int X = analogRead(pinX);
    // in steps of 1 degree
    myservo1.write(pos1);              // tell servo to go to position in variable 'pos'
    delay(15);                        // waits 15ms for the servo to reach the position
    pos1=pos1+1; 
    if(X>450){
      break;
      }
    }
  }

//ANTI-CLOCKWISE MOVEMENT OF SERVO
void servo_mov_2(){
    while (pos1 >= 0) { // goes from 180 degrees to 0 degrees
    int X = analogRead(pinX);
    myservo1.write(pos1);              // tell servo to go to position in variable 'pos'
    delay(15);                       // waits 15ms for the servo to reach the position
    pos1=pos1-1;
    if(X<570){
      break;
      }
    }
}

//CLOSING OF CLAW
void servo_claw_1(){
  while (pos2 <= 60) {              // goes from 0 degrees to 180 degrees
    int Y = analogRead(pinY);
    // in steps of 1 degree
    myservo2.write(pos2);         // tell servo to go to position in variable 'pos'
    delay(15);                   // waits 15ms for the servo to reach the position
    pos2=pos2+1;
    if(Y > 450){
      break; 
      }
    }
  }

//OPENING OF CLAW
void servo_claw_2(){
    while (pos2 >= 0) {               // goes from 18U0 degrees to 0 degrees
    int Y = analogRead(pinY);
    myservo2.write(pos2);              // tell servo to go to position in variable 'pos'
    delay(15);                       // waits 15ms for the servo to reach the position
    pos2=pos2-1;
    if(Y < 570){
      break;
      }
    }
  }
