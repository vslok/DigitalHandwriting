using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models; // For Method enum
using DigitalHandwriting.Helpers; // For Calculations
using DigitalHandwriting.Models; // For User and AuthenticationResult
using System;
using System.Collections.Generic;
using System.Linq; // Required for .Any()
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public static class AuthenticationService
    {
        public static bool PasswordAuthentication(User user, string password)
        {
            if (user == null)
            {
                // Consider logging this critical error as well
                // Console.WriteLine("Error: User object is null for password authentication.");
                throw new ArgumentNullException(nameof(user), "User cannot be null for password authentication.");
            }

            var userPasswordHash = user.Password;
            var userPasswordSalt = user.Salt;

            if (string.IsNullOrEmpty(userPasswordSalt))
            {
                Console.WriteLine($"Error: Salt is missing for user {user.Login}. Cannot authenticate password.");
                return false; // Or throw InvalidOperationException
            }
            if (string.IsNullOrEmpty(userPasswordHash))
            {
                Console.WriteLine($"Error: Password hash is missing for user {user.Login}. Cannot authenticate password.");
                return false; // Or throw InvalidOperationException
            }


            var inputPasswordHash = EncryptionService.GetPasswordHash(password, userPasswordSalt);
            return userPasswordHash.Equals(inputPasswordHash);
        }

        public static async Task<AuthenticationResult> HandwritingAuthentication(
            User user,
            List<double> loginKeyPressedTimes,
            List<double> loginBetweenKeysTimes,
            Method authMethod = Method.FilteredManhattan)
        {
            if (user == null)
            {
                throw new ArgumentNullException(nameof(user), "User cannot be null for handwriting authentication.");
            }

            var hUserProfile = user.HSampleValues;
            var duUserProfile = user.UDSampleValues;

            // Handle cases where the profile data itself might be missing or fundamentally unusable.
            // If HSampleValues or UDSampleValues are null on the User object,
            // this indicates a data integrity problem or an issue during user loading/creation.
            if (hUserProfile == null || duUserProfile == null)
            {
                Console.WriteLine($"Error: User {user.Login} has null HSampleValues or UDSampleValues. Handwriting authentication cannot proceed with this user profile.");
                // The specific IAuthenticationMethod (e.g., FilteredManhattan) should ideally be robust enough
                // to handle empty or invalid profile data passed to its constructor or Authenticate method.
                // It should return an AuthenticationResult indicating IsAuthenticated = false.
                // We will pass empty lists to CalculateMeanValue and then to the factory,
                // relying on downstream components to handle this "unusable profile" state.
                hUserProfile = new List<List<double>>();
                duUserProfile = new List<List<double>>();
            }

            // It's also possible that HSampleValues/UDSampleValues are not null but contain no actual data points.
            // Example: List is not null, but contains 0 lists, or contains only empty inner lists.
            // CalculateMeanValue should be robust to this. If it isn't, it might throw an exception here.
            // If it returns NaN or an empty list for medians, the authentication method should handle that.
            // No explicit check for !Any(list => list.Any()) here, assuming CalculateMeanValue and the auth method are robust.

            var hUserMedian = Calculations.CalculateMeanValue(hUserProfile); // Must handle empty or list of empty lists
            var udUserMedian = Calculations.CalculateMeanValue(duUserProfile); // Must handle empty or list of empty lists

            // The AuthenticationMethodFactory or the IAuthenticationMethod itself should handle cases where
            // profiles or medians are "empty" or indicate unusable data (e.g., medians are NaN or empty lists).
            // The factory might return a specific "no-op" or "always fail" authenticator,
            // or the authenticator's Authenticate method will return IsAuthenticated = false.
            var authenticationMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                authMethod,
                hUserMedian,
                udUserMedian,
                hUserProfile,
                duUserProfile);

            return await authenticationMethod.Authenticate(user.Login, 1, loginKeyPressedTimes, loginBetweenKeysTimes);
        }
    }
}
