using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using System.Windows.Markup;

namespace DigitalHandwriting.Converters
{
    class RegistrationInformationStringConverter : MarkupExtension, IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var step = (int)value;
            return step switch
            {
                0 => ConstStrings.FirstRegistrationStepDescription,
                1 => ConstStrings.SecondRegistrationStepDescription,
                2 => ConstStrings.ThirdRegistrationStepDescription,
                3 => ConstStrings.FourthRegistrationStepDescription,
                4 => ConstStrings.FifthRegistrationStepDescription,
                _ => " "
            };
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }

        public override object ProvideValue(IServiceProvider serviceProvider)
        {
            return this;
        }
    }
}
