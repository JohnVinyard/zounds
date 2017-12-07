from io import BytesIO


class PreDownload(BytesIO):
    def __init__(self, initial_bytes, url):
        super(PreDownload, self).__init__(initial_bytes)
        if not url:
            raise ValueError('url must be provided')
        self.url = url
