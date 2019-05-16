

import java.io.*;
import java.util.Scanner;

public class Decode {
    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(new File("D://3052018041王法磊.daz"));
        FileOutputStream output = new FileOutputStream("D://3052018041王法磊.txt");
        String str = "";
        while(scanner.hasNext()){
            str += (char)Integer.valueOf(scanner.next(), 10).intValue();
        }
        output.write(str.getBytes("unicode"));
    }
}
