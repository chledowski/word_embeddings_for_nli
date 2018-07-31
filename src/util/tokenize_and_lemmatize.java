import edu.stanford.nlp.simple.*;
import java.io.*;
import java.util.*;

public class tokenize_and_lemmatize {
    public static void main(String[] args) throws FileNotFoundException{ 
        // Open the file
        try{
            Scanner scan = new Scanner(new File(args[0]));
            PrintWriter writer = new PrintWriter(args[1], "UTF-8");

            while(scan.hasNextLine()){
                String line = scan.nextLine();
                // System.out.println(line);
                Sentence sent = new Sentence(line);
                List<String> lemmas = sent.lemmas();
                String joined2 = String.join(" ", lemmas);
                writer.println(joined2);
            }
            writer.close();

        } catch (Exception e) {
           e.printStackTrace();
        }
    }
}