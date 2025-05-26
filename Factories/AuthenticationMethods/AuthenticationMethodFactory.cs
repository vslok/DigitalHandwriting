using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods
{
    public enum Method
    {
        Euclidian,
        NormalizedEuclidian,
        Manhattan,
        FilteredManhattan,
        ScaledManhattan,
        ITAD,
        CNN,
        GRU,
        KNN,
        LSTM,
        MLP,
        NaiveBayes,
        RandomForest,
        SVM,
        XGBoost,
    }

    public static class AuthenticationMethodFactory
    {
        public static AuthenticationMethod GetAuthenticationMethod(
                Method method,
                List<double> userKeyPressedTimes,
                List<double> userBetweenKeysTimes,
                List<List<double>> userKeyPressedTimesProfile,
                List<List<double>> userBetweenKeysTimesProfile
            )
        {
            switch (method)
            {
                case Method.Euclidian:
                    return new EuclidianAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.NormalizedEuclidian:
                    return new NormalizedEuclidianAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.Manhattan:
                    return new ManhattanAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.FilteredManhattan:
                    return new FilteredManhattanAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.ScaledManhattan:
                    return new ScaledManhattanAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.ITAD:
                    return new ITADAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.CNN:
                    return new CNNAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.GRU:
                    return new GRUAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.KNN:
                    return new KNNAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.LSTM:
                    return new LSTMAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.MLP:
                    return new MLPAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.NaiveBayes:
                    return new NaiveBayesAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.RandomForest:
                    return new RandomForestAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.SVM:
                    return new SVMAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                case Method.XGBoost:
                    return new XGBoostAuthenticationMethod(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile);
                default:
                    throw new ApplicationException("Authentication method not supported: " + method.ToString());
            }
        }
    }
}
