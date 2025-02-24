﻿using DigitalHandwriting.Factories.AuthenticationMethods.Models;
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
                default:
                    throw new ApplicationException("Authentication method not supported");
            }
        }
    }
}
