﻿using DigitalHandwriting.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public static class AuthentificationService
    {
        public static bool HandwritingAuthentification(User user, List<int> loginKeyPressedTimes, List<int> loginBeetwenKeysTimes, 
            out double keyPressedDistance, out double beetweenKeysDistance)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimes);
            var userBeetwenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BeetwenKeysTimes);

            keyPressedDistance = EuclideanDistance(userKeyPressedTimes, loginKeyPressedTimes);
            beetweenKeysDistance = EuclideanDistance(userBeetwenKeysTimes, loginBeetwenKeysTimes);

            return keyPressedDistance <= 0.20 && beetweenKeysDistance <= 0.30;
        }
        private static double EuclideanDistance(List<int> etVector, List<int> curVector)
        {
            var normalizedEtVector = Normalize(etVector);
            var normalizedCurVector = Normalize(curVector);

            var sum = 0.0;
            for (int i = 0; i < normalizedEtVector.Count; i++)
            {
                sum += Math.Pow(normalizedEtVector.ElementAtOrDefault(i) - normalizedCurVector.ElementAtOrDefault(i), 2);
            }
            return Math.Round(Math.Sqrt(sum), 2);
        }

        private static List<double> Normalize(List<int> vector)
        {
            var distance = Math.Sqrt(vector.ConvertAll(el => Math.Pow(el, 2)).Sum());
            return vector.ConvertAll(el => el / distance);
        }
    }
}