namespace ONNXBayes

{
    // Testing the Naive Bayes model using ONNX in C#
    public class Program
    {
        static void Main()
        {
            if (!File.Exists("Model/Model.onnx"))
            {
                Console.WriteLine("Model file not found.");
                return;
            }
            
            StartingNaiveBayes();
        }

        static void StartingNaiveBayes()
        {



        }
    }
}