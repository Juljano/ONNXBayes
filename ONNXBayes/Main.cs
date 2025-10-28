using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ONNXBayes
{
    // Testing the Naive Bayes model using ONNX in C#
    public class Program
    {
        private static string _modelPath = "Model/Naive-Bayes-Model.onnx";


        static void Main()
        {
            if (!File.Exists(_modelPath))
            {
                Console.WriteLine("Model file not found.");
                return;
            }

            StartingNaiveBayes();
        }

        static void StartingNaiveBayes()
        {
            using var session = new InferenceSession(_modelPath);

            Console.WriteLine("Please enter a message to rate:");
            string[] message = {Console.ReadLine()};
            Console.WriteLine($"Your Message: {message[0]}");

            var inputTensor = new DenseTensor<string>(message, new int[] { message.Length, 1 });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = session.Run(inputs);

            var labelTensor = results.First(r => r.Name == "output_label").AsTensor<string>();

            foreach (var prediction in labelTensor)
            {
                
                switch (prediction)
                {
                    case "spam": 
                        Console.WriteLine($"Prediction: {prediction} - Diese Nachricht ist Spam - Bitte diese Nachricht löschen");
                        break;
                    case "ham":
                        Console.WriteLine($"Prediction: {prediction} - Diese Nachricht ist kein Spam - Du kannst die Nachricht behalten");
                        break;
                    default:
                        Console.WriteLine("Diese Nachricht kann nicht eingeschätzt werden");
                        break;
                }
               
            }
        }
    }
}
