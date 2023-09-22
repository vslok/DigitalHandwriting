using DigitalHandwriting.Context;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Stores;
using DigitalHandwriting.Models;
using DigitalHandwriting.Views;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Security;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Input;
using DigitalHandwriting.Commands;
using DigitalHandwriting.Services;
using Microsoft.Extensions.Logging;

namespace DigitalHandwriting.ViewModels
{
    public class HomeViewModel : BaseViewModel
    {
        public ICommand OnRegistrationButtonClickCommand { get; set; }
        public ICommand OnAuthentificationButtonClickCommand { get; set; }
        public ICommand OnCheckTextBoxKeyDownEventCommand { get; set; }
        public ICommand OnCheckTextBoxKeyUpEventCommand { get; set; }

        private int _checkTextCurrentLetterIndex = 0;

        private bool _isAuthentificationButtonEnabled = false;

        private Helper _helper = new Helper();

        private string _checkTextWithUpperCase => ConstStrings.CheckText.ToUpper();

        private string _userCheckText = "";

        private string _userLogin = "";

        private DateTime _lastKeyDownTime;

        private List<int> _keyPressedTimes = new List<int>();

        private List<int> _beetwenKeysTimes = new List<int>();

        private ApplicationContext db = new ApplicationContext();

        public HomeViewModel(
            Func<RegistrationViewModel> registrationViewModelFactory,
            NavigationStore navigationStore)
        {
            OnRegistrationButtonClickCommand = new NavigateCommand<RegistrationViewModel>(
                new NavigationService<RegistrationViewModel>(
                    navigationStore,
                    () => registrationViewModelFactory()));

            OnAuthentificationButtonClickCommand = new Command(OnAuthentificationButtonClick);
            OnCheckTextBoxKeyDownEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyDownEvent);
            OnCheckTextBoxKeyUpEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyUpEvent);

            db.Database.EnsureCreated();
        }

        public bool IsAuthentificationButtonEnabled
        {
            get => _isAuthentificationButtonEnabled;
            set => SetProperty(ref _isAuthentificationButtonEnabled, value);
        }

        public string UserLogin
        {
            get => _userLogin;
            set
            {
                SetProperty(ref _userLogin, value);
            }
        }

        public string UserCheckText
        {
            get => _userCheckText;
            set => SetProperty(ref _userCheckText, value);
        }

        private void OnCheckTextBoxKeyUpEvent(object props)
        {
            var e = (KeyEventArgs)props;

            if (!_helper.CheckCurrentLetterKeyPressed(e, _checkTextCurrentLetterIndex - 1, _checkTextWithUpperCase))
            {
                e.Handled = true;
                return;
            }

            var keyPressedTime = (DateTime.UtcNow - _lastKeyDownTime).Milliseconds;
            _keyPressedTimes.Add(keyPressedTime);

            if (_checkTextCurrentLetterIndex == _checkTextWithUpperCase.Length)
            {
                db.Users.Load();

                var userRegistrated = db.Users.Local.Select(user => user.Login).Contains(_userLogin);
                if (!userRegistrated)
                {
                    ResetWindowState();
                }
                else
                {
                    IsAuthentificationButtonEnabled = true;
                }
            }
        }

        private void OnCheckTextBoxKeyDownEvent(object props)
        {
            var e = (KeyEventArgs)props;
            if (!_helper.CheckCurrentLetterKeyPressed(e, _checkTextCurrentLetterIndex, _checkTextWithUpperCase))
            {
                e.Handled = true;
                return;
            }

            if (!string.IsNullOrEmpty(_userCheckText))
            {
                var time = (DateTime.UtcNow - _lastKeyDownTime).Milliseconds;
                _beetwenKeysTimes.Add(time);
            }

            _lastKeyDownTime = DateTime.UtcNow;
            _checkTextCurrentLetterIndex++;
        }

        private void OnRegistrationButtonClick()
        {
            //var window = new Registration();
            //window.ShowDialog();
        }

        private void OnAuthentificationButtonClick()
        {
            var user = db.Users.Select(user => user).Where(user => user.Login == UserLogin).FirstOrDefault();
            //var window = new UserInfo(user, _keyPressedTimes, _beetwenKeysTimes);
            //window.ShowDialog();
            ResetWindowState();
        }

        private void ResetWindowState()
        {
            IsAuthentificationButtonEnabled = false;
            UserCheckText = "";
            _checkTextCurrentLetterIndex = 0;

            _keyPressedTimes = new List<int>();
            _beetwenKeysTimes = new List<int>();
        }

    }
}
