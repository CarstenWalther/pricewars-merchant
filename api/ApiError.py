class ApiError(Exception):
    def __init__(self, status_code, url, text):
        self.status_code = status_code
        self.url = url
        self.text = text

    def __str__(self):
        return '\nStatus code: {}\nURL: {}\nText: {}'.format(self.status_code, self.url, self.text)
