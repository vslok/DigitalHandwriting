using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public static class AuthentificationService
    {
        public static bool PasswordAuthentification(User user, string password)
        {
            var userPasswordHash = user.Password;
            var userPasswordSalt = user.Salt;

            var inputPasswordHash = EncryptionService.GetPasswordHash(password, userPasswordSalt);

            return userPasswordHash.Equals(inputPasswordHash);
        }

        public static bool HandwritingAuthentification(User user, List<int> loginKeyPressedTimes, List<int> loginBetweenKeysTimes,
            out double keyPressedDistance, out double betweenKeysDistance)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimes);
            var userBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimes);

            var userKeyPressedNormalized = Calculations.Normalize(userKeyPressedTimes);
            var loginKeyPressedNormalized = Calculations.Normalize(loginKeyPressedTimes);

            var keyPressedDistanceError = 0;            
            for (int i = 0; i < userKeyPressedNormalized.Count; i++)
            {
                var result = Calculations.ManhattanDistance(userKeyPressedNormalized[i], loginKeyPressedNormalized[i]);
                if (result > 0.2)
                {
                    keyPressedDistanceError++;
                }
            }

            var userBetweenKeysNormalized = Calculations.Normalize(userBetweenKeysTimes);
            var loginBetweenKeysNormalized = Calculations.Normalize(loginBetweenKeysTimes);

            var keyBetweenKeysDistanceError = 0;
            for (int i = 0; i < userBetweenKeysNormalized.Count; i++)
            {
                var result = Calculations.ManhattanDistance(userBetweenKeysNormalized[i], loginBetweenKeysNormalized[i]);
                if (result > 0.2)
                {
                    keyBetweenKeysDistanceError++;
                }
            } 

            keyPressedDistance = keyPressedDistanceError / (double)userKeyPressedTimes.Count;
            betweenKeysDistance = keyBetweenKeysDistanceError / (double)userBetweenKeysTimes.Count;

            var authResult = (keyPressedDistanceError + keyBetweenKeysDistanceError) / (double)(userKeyPressedTimes.Count + userBetweenKeysTimes.Count);

            Trace.WriteLine($"Key pressed errors: {keyPressedDistanceError}, Between keys errors: {keyBetweenKeysDistanceError}, auth result : {authResult}");

            //keyPressedDistance = Calculations.ManhattanDistance(userKeyPressedTimes, loginKeyPressedTimes);
            //betweenKeysDistance = Calculations.ManhattanDistance(userBetweenKeysTimes, loginBetweenKeysTimes);

            return authResult < 0.15;
        }
    }
}
