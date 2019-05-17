public class Test {
    public static void main(String[] args) {
        Man man = new Child();
        System.out.println(man.age);
        man.say();
    }
}

class Man {
    int age = 0;
    public void say(){
        System.out.println("man");
    }
}

class Child extends  Man{
    int age = 1;

    public void say(){
        System.out.println("child");
    }

}