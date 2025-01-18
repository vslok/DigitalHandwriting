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
    public class HandwritingAuthenticationResult
    {
        public double KeyPressedDistance { get; set; }

        public double BetweenKeysDistance { get; set; }

        public double BetweenKeysPressDistance { get; set; }

        public double AuthenticationResult { get; set; }
    }

    public static class AuthenticationService
    {
        public static bool PasswordAuthentication(User user, string password)
        {
            var userPasswordHash = user.Password;
            var userPasswordSalt = user.Salt;

            var inputPasswordHash = EncryptionService.GetPasswordHash(password, userPasswordSalt);

            return userPasswordHash.Equals(inputPasswordHash);
        }

        public static bool HandwritingAuthentication(
            User user, 
            List<int> loginKeyPressedTimes, 
            List<int> loginBetweenKeysTimes,
            List<int> loginBetweenKeysPressTimes,
            out double keyPressedDistance, 
            out double betweenKeysDistance,
            out double betweenKeysPressDistance)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimes);
            var userBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimes);
            var userBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimes);

            if (userKeyPressedTimes == null || userBetweenKeysTimes == null || userBetweenKeysPressTimes == null)
            {
                throw new Exception("Incorrect user authentication parameters in db");
            }

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

            var userBetweenKeysPressNormalized = Calculations.Normalize(userBetweenKeysPressTimes);
            var loginBetweenKeysPressNormalized = Calculations.Normalize(loginBetweenKeysPressTimes);

            var keyBetweenKeysPressDistanceError = 0;
            for (int i = 0; i < userBetweenKeysPressNormalized.Count; i++)
            {
                var result = Calculations.ManhattanDistance(userBetweenKeysPressNormalized[i], loginBetweenKeysPressNormalized[i]);
                if (result > 0.2)
                {
                    keyBetweenKeysPressDistanceError++;
                }
            }

            keyPressedDistance = keyPressedDistanceError / (double)userKeyPressedTimes.Count;
            betweenKeysDistance = keyBetweenKeysDistanceError / (double)userBetweenKeysTimes.Count;
            betweenKeysPressDistance = keyBetweenKeysPressDistanceError / (double)userBetweenKeysPressTimes.Count;

            var authResult = (keyPressedDistanceError + keyBetweenKeysDistanceError + keyBetweenKeysPressDistanceError) 
                / (double)(userKeyPressedTimes.Count + userBetweenKeysTimes.Count + userBetweenKeysPressTimes.Count);

            Trace.WriteLine($"Key pressed errors: {keyPressedDistanceError}, " +
                $"Between keys errors: {keyBetweenKeysDistanceError}, " +
                $"Between keys press errors: {keyBetweenKeysPressDistanceError} " +
                $"auth result : {authResult}");

            //keyPressedDistance = Calculations.ManhattanDistance(userKeyPressedTimes, loginKeyPressedTimes);
            //betweenKeysDistance = Calculations.ManhattanDistance(userBetweenKeysTimes, loginBetweenKeysTimes);

            return authResult < 0.15;
        }
    }
}
