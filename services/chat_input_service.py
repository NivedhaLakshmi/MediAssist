import base64
from io import BytesIO


class ComponentUploadedFile(BytesIO):
    """
    File-like object for files received from
    the custom Streamlit chat component.
    """

    def __init__(
        self,
        data,
        name,
        file_type="application/pdf"
    ):
        super().__init__(data)

        self.name = name
        self.type = file_type
        self.size = len(data)


def convert_component_file(file_data):

    file_bytes = base64.b64decode(
        file_data["data"]
    )

    return ComponentUploadedFile(
        data=file_bytes,
        name=file_data["name"],
        file_type=file_data.get(
            "type",
            "application/pdf"
        )
    )


def convert_component_files(files):

    uploaded_files = []

    for file_data in files:

        uploaded_file = convert_component_file(
            file_data
        )

        uploaded_files.append(
            uploaded_file
        )

    return uploaded_files