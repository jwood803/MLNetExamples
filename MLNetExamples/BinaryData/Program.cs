using Microsoft.ML;
using System.IO;

namespace BinaryData
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            using (var stream = new FileStream("./housing_binary.idv", FileMode.Create))
            {
                context.Data.SaveAsBinary(data, stream);
            }

            var binaryData = context.Data.LoadFromBinary("./housing_binary.idv");

            var preview = binaryData.Preview();
        }
    }
}
