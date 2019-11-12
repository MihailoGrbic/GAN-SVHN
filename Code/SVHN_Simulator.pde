PFont fonts[]=new PFont[100];
String[] fontList = PFont.list();
int count=-1;
int num=0;

void setup() {
  size(32,32);
  for(int i=0;i<100;i++)
  {
    fonts[i] = createFont(fontList[i],16,true);
  }
}

int[] colour(String A)
{
  int x[]=new int[3];

  x[0]=unhex(A.substring(4,6));
  x[1]=unhex(A.substring(2,4));
  x[2]=unhex(A.substring(2));
  return x;
}

void draw() {
  if(count==20000)
  {
    num++;
    count=-1;
    if(num==10)exit();
  }
  count++;
  float rot=random(-0.1,0.1);
  rotate(rot);

  int rbg=(int)random(256); int gbg=(int)random(256); int bbg=(int)random(256);background(rbg,gbg,bbg);


  int rfont=(int)random(256); int gfont=(int)random(256); int bfont=(int)random(256);
  while((rfont-rbg)*(rfont-rbg)<256)
  {
    rfont=(int)random(256);
  }
  while((bfont-rbg)*(bfont-rbg)<256)
  {
    bfont=(int)random(256);
  }
  while((gfont-gbg)*(gfont-gbg)<256)
  {
    gfont=(int)random(256);
  }
  fill(rfont, gfont, bfont);

  int size=(int)random(28,40);
  int font=(int)random(40);
  textFont(fonts[font],size);

  //num=(int)random(10);
  
  int x=(int)random(5,10);
  int y=32-(int)random(0,40-size);
  
  fill(rfont, gfont, bfont);
  text(num,x,y);
 
  
  float distractorDigitL=random(1);
  if(distractorDigitL<0.3)
  {
    int numd=(int)random(10);
    int xL=x-(int)random(18,25);
    int yL=y-(int)random(-2,2);
    text(numd,xL,yL);
  }
  
  float distractorDigitR=random(1);
  if(distractorDigitR<0.3)
  {
    int numd=(int)random(10);
    int xL=x+(int)random(18,22);
    int yL=y-(int)random(-2,2);
    text(numd,xL,yL);
  }
  filter(BLUR);
  filter(BLUR);
  filter(BLUR);


  
  colorMode(HSB);
  
  
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 32; j++) {
      color t = get(i, j);
      float tb = brightness(t);
      tb += noise(i, j) * 10 + noise(0.06*i, 0.06*j) * 100;
      float th = hue(t) + random(-5, 5);
      float ts = saturation(t) + noise(0.1*i, 0.1*j) * 20;
      set(i, j, color((int) th, (int) ts, (int) tb));
    }
  }
  
  saveFrame("line-"+num+"-"+count+".png"); 
  
}
