namespace DeepNeuralNetworkUpdate
{
    public class ImageModelInput
    {
        public byte[] Image { get; set; }

        public uint LabelAsKey { get; set; }

        public string ImagePath { get; set; }

        public string Label { get; set; }
    }
}
