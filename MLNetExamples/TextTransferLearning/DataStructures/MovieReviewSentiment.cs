using Microsoft.ML.Data;

namespace TextTransferLearning.DataStructures
{
    public class MovieReviewSentiment
    {
        [VectorType(2)]
        public float[] Prediction { get; set; }
    }
}
