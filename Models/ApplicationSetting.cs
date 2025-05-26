using System.ComponentModel.DataAnnotations;

namespace DigitalHandwriting.Models
{
    public class ApplicationSetting
    {
        [Key]
        public string Key { get; set; }

        public string Value { get; set; }
    }
}
