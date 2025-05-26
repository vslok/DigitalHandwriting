using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using System.Collections.Generic;

namespace DigitalHandwriting.Context
{
    public static class ApplicationConfiguration
    {
        public static List<Method> ValidationAuthenticationMethods { get; set; } = new List<Method>
        {
            Method.Euclidian,
            Method.NormalizedEuclidian,
            Method.Manhattan,
            Method.FilteredManhattan,
            Method.ScaledManhattan,
            Method.ITAD,
            Method.CNN,
            Method.GRU,
            Method.KNN,
            Method.LSTM,
            Method.MLP,
            Method.NaiveBayes,
            Method.RandomForest,
            Method.SVM,
            Method.XGBoost
        };

        public static Method DefaultAuthenticationMethod { get; set; } = Method.FilteredManhattan;

        public static int RegistrationPassphraseInputs { get; set; } = 3; // Default value, can be changed
    }
}
