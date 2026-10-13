import unittest

from serverless.credentials import parse_credentials_file
from serverless.vision import Person, _retry_after_s, count_people, parse_analysis


class ParseAnalysisTests(unittest.TestCase):
    def test_reads_people_boxes_and_image_size(self):
        analysis = parse_analysis(
            {
                "metadata": {"width": 1920, "height": 1080},
                "peopleResult": {
                    "values": [
                        {"boundingBox": {"x": 10, "y": 20, "w": 30, "h": 60}, "confidence": 0.66},
                        {"boundingBox": {"x": 5, "y": 6, "w": 7, "h": 8}, "confidence": 0.02},
                    ]
                },
            }
        )

        self.assertEqual((analysis.width, analysis.height), (1920, 1080))
        self.assertEqual(analysis.people[0], Person(x=10, y=20, w=30, h=60, confidence=0.66))
        self.assertEqual(len(analysis.people), 2)

    def test_counts_only_people_above_the_confidence_threshold(self):
        people = [Person(0, 0, 1, 1, confidence) for confidence in (0.66, 0.5, 0.27, 0.01)]

        self.assertEqual(count_people(people), 2)
        self.assertEqual(count_people(people, min_confidence=0.2), 3)

    def test_retry_after_falls_back_on_missing_or_invalid_header(self):
        self.assertEqual(_retry_after_s("5"), 5.0)
        self.assertEqual(_retry_after_s(None), 2.0)
        self.assertEqual(_retry_after_s("Wed, 21 Oct 2026 07:28:00 GMT"), 2.0)


class CredentialsFileTests(unittest.TestCase):
    def test_keys_are_scoped_by_section_and_values_keep_equals_signs(self):
        values = parse_credentials_file(
            "# Storage account\n"
            "Key=storage-key\n"
            "ConnectionString=DefaultEndpointsProtocol=https;AccountKey=abc==\n"
            "\n"
            "# Computer Vision resource\n"
            "Key 1=vision-key\n"
            "Endpoint=https://example.cognitiveservices.azure.com/\n"
        )

        self.assertEqual(values[("storage account", "key")], "storage-key")
        self.assertEqual(
            values[("storage account", "connectionstring")],
            "DefaultEndpointsProtocol=https;AccountKey=abc==",
        )
        self.assertEqual(values[("computer vision resource", "key 1")], "vision-key")


if __name__ == "__main__":
    unittest.main()
