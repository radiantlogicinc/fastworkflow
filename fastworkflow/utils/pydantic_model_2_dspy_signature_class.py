from typing import Annotated, Type, Union, get_args, get_origin

import dspy
from dspy import InputField, OutputField, Signature
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


class TypedPredictorSignature:
    @classmethod
    def create(
        cls,
        pydantic_class_for_dspy_input_fields: Type[BaseModel],
        pydantic_class_for_dspy_output_fields: Type[BaseModel],
        prefix_instructions: str = "",
    ) -> Type[dspy.Signature]:
        """
        Return a DSPy Signature class that can be used to extract the output parameters.

        :param pydantic_class_for_dspy_input_fields: Pydantic class that defines the DSPy InputField's.
        :param pydantic_class_for_dspy_output_fields: Pydantic class that defines the DSPy OutputField's.
        :param prefix_instructions: Optional text that is prefixed to the instructions.
        :return: A DSPy Signature class optimizedfor use with a TypedPredictor to extract structured information.
        """
        instructions = (
            "Use only the available information to extract the output fields.\n\n"
        )
        if prefix_instructions:
            prefix_instructions += "\n\n"
            instructions = prefix_instructions + instructions

        dspy_fields = {}
        for (
            field_name,
            field,
        ) in pydantic_class_for_dspy_input_fields.model_fields.items():
            if field.default and "typing.Annotated" in str(field.default):
                raise ValueError(
                    f"Field '{field_name}' is annotated incorrectly. See 'Constraints on compound types' in https://docs.pydantic.dev/latest/concepts/fields/"
                )

            is_default_value_specified, is_marked_as_optional, inner_field = (
                cls._process_field(field)
            )
            if is_marked_as_optional:
                if field.default is None or field.default is PydanticUndefined:
                    field.default = "null"
                field.description = inner_field.description
                field.examples = inner_field.examples
                field.metadata = inner_field.metadata
                field.json_schema_extra = inner_field.json_schema_extra
            else:
                field.validate_default = False

            input_field = dspy.InputField(desc=field.description)
            dspy_fields[field_name] = (field.annotation, input_field)

        for (
            field_name,
            field,
        ) in pydantic_class_for_dspy_output_fields.model_fields.items():
            if field.default and "typing.Annotated" in str(field.default):
                raise ValueError(
                    f"Field '{field_name}' is annotated incorrectly. See 'Constraints on compound types' in https://docs.pydantic.dev/latest/concepts/fields/"
                )

            is_default_value_specified, is_marked_as_optional, inner_field = (
                cls._process_field(field)
            )
            if is_marked_as_optional:
                if field.default is None or field.default is PydanticUndefined:
                    field.default = "null"
                field.description = inner_field.description
                field.examples = inner_field.examples
                field.metadata = inner_field.metadata
                field.json_schema_extra = inner_field.json_schema_extra
            else:
                field.validate_default = False

            if field.default is PydanticUndefined:
                raise ValueError(
                    f"Field '{field_name}' has no default value. Required fields must have a default value. "
                    "Change the field to be Optional or specify a default value."
                )

            output_field = dspy.OutputField(
                desc=field.description or ""
            )
            dspy_fields[field_name] = (field.annotation, output_field)

            instructions += f"When extracting '{field_name}':\n"
            instructions += f"If it is not mentioned in the input fields, return: '{field.default}'. "

            if examples := field.examples:
                quoted_examples = [f"'{example}'" for example in examples]
                instructions += (
                    f"Example values of {field_name} are: {', '.join(quoted_examples)} etc. "
                )

            if field.metadata:
                constraints = [
                    meta for meta in field.metadata if "Validator" not in str(meta)
                ]
                if (
                    field.json_schema_extra
                    and "invalid_value" in field.json_schema_extra
                ):
                    instructions += f"If the extracted value does not conform to: {constraints}, return: '{field.json_schema_extra['invalid_value']}'."
                else:
                    print(
                        f"WARNING - Field: '{field_name}' is missing an 'invalid_value' attribute. Fields with value constraints should specify an 'invalid_value'."
                    )
                    instructions += f"If the extracted value does not conform to: {constraints}, return: '{field.default}'."

            instructions += "\n\n"

        return dspy.Signature(dspy_fields, instructions.strip())

    @classmethod
    def _process_field(cls, field: FieldInfo) -> tuple[bool, bool, FieldInfo]:
        is_default_value_specified = not field.is_required()
        is_marked_as_optional, inner_type, field_info = cls._analyze_field_annotation(
            field.annotation
        )
        if field_info:
            field_info.annotation = inner_type
            return is_default_value_specified, is_marked_as_optional, field_info
            # if field_info.json_schema_extra and 'not_found_value' in field_info.json_schema_extra:
            #     field_info.default = field_info.json_schema_extra['not_found_value']

        return is_default_value_specified, is_marked_as_optional, field

    @classmethod
    def _analyze_field_annotation(cls, annotation):
        is_optional = False
        inner_type = annotation
        field_info = None

        # If field is specfied as Optional[Annotated[...]]
        if get_origin(annotation) is Union:
            args = get_args(annotation)
            if type(None) in args:
                is_optional = True
                inner_type = args[0] if args[0] is not type(None) else args[1]
        # Not sure why I added this, perhaps for some other way of specifying optional fields?
        # elif hasattr(annotation, '_name') and annotation._name == 'Optional':
        #     is_optional = True
        #     inner_type = get_args(annotation)[0]

        # Check if it's Annotated
        if get_origin(inner_type) is Annotated:
            args = get_args(inner_type)
            inner_type = args[0]
            for arg in args[1:]:
                if isinstance(arg, FieldInfo):
                    field_info = arg
                    break

        return is_optional, inner_type, field_info


# if __name__ == "__main__":
#     class CommandExtractionInput(BaseModel):
#         command: str

#     class PydanticOutput1(BaseModel):
#         @field_validator("name", mode="wrap")
#         @staticmethod
#         def validate_name(name, handler):
#             try:
#                 return handler(name)
#             except ValidationError:
#                 return 'INVALID'

#         name: Annotated[str,
#                         Field(default='NOT_FOUND', max_length=15,
#                             title='Name', description='The name of the person',
#                             examples=['John Doe', 'Jane Doe'],
#                             json_schema_extra={'invalid_value': 'INVALID'}
#                             )
#                     ]
#     dspy_signature_class1 = TypedPredictorSignature.create(
#         CommandExtractionInput, PydanticOutput1)

#     class PydanticOutput2(BaseModel):
#         @field_validator("age", mode="wrap")
#         @staticmethod
#         def validate_age(age, handler):
#             try:
#                 return handler(age)
#             except ValidationError:
#                 return -8888

#         age: Annotated[int,
#                     Field(gt=0, lt=150, default=-999,
#                             json_schema_extra={'invalid_value': '-8888'}
#                             )
#                     ]
#     dspy_signature_class2 = TypedPredictorSignature.create(
#         CommandExtractionInput, PydanticOutput2)

#     class PydanticOutput3(BaseModel):
#         age: Annotated[int,
#                     Field(gt=0, lt=150,
#                             json_schema_extra={'invalid_value': '-8888'}
#                             )
#                     ] = -999
#     dspy_signature_class3 = TypedPredictorSignature.create(
#         CommandExtractionInput, PydanticOutput3)

#     class PydanticOutput4(BaseModel):
#         age: Optional[Annotated[int,
#                     Field(gt=0, lt=150, default="null",
#                             json_schema_extra={'invalid_value': '-8888'}
#                             )]]
#     dspy_signature_class4 = TypedPredictorSignature.create(
#         CommandExtractionInput, PydanticOutput4)

#     class PydanticOutput5(BaseModel):
#         @field_validator("email", mode="wrap")
#         @staticmethod
#         def validate_email(email, handler):
#             try:
#                 return handler(email)
#             except ValidationError:
#                 return 'INVALID'

#         email: Annotated[str,
#                         Field(default='NOT_FOUND',
#                             pattern=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$',
#                             json_schema_extra={'invalid_value': 'INVALID'}
#                             )
#                         ]
#     dspy_signature_class5 = TypedPredictorSignature.create(
#         CommandExtractionInput, PydanticOutput5)

#     class PydanticOutput6(BaseModel):
#         age: int = Field(gt=0, lt=150, default=-999,
#                          json_schema_extra={'invalid_value': '-8888'}
#                         )
#     dspy_signature_class6 = TypedPredictorSignature.create(
#         CommandExtractionInput, PydanticOutput6)

#     dspy_lm = DSPyUtils.get_task_lm()
#     with dspy.context(lm=dspy_lm):
#         extract_cmd_params1 = dspy.TypedChainOfThought(
#             dspy_signature_class1)
#         extract_cmd_params2 = dspy.TypedChainOfThought(
#             dspy_signature_class2)
#         extract_cmd_params3 = dspy.TypedChainOfThought(
#             dspy_signature_class3)
#         extract_cmd_params4 = dspy.TypedChainOfThought(
#             dspy_signature_class4)
#         extract_cmd_params5 = dspy.TypedChainOfThought(
#             dspy_signature_class5)

#         input_for_parameter_extraction = CommandExtractionInput(
#             # command = "A random command."
#             # command = "My name is kjhd and I am 200 years old. My email is 9236"
#             command = "Hello, my name is John Doe and I am 25 years old. My email is john.doe@example.com."
#         )

#         prediction1 = extract_cmd_params1(**input_for_parameter_extraction.model_dump())
#         prediction2 = extract_cmd_params2(**input_for_parameter_extraction.model_dump())
#         prediction3 = extract_cmd_params3(**input_for_parameter_extraction.model_dump())
#         prediction4 = extract_cmd_params4(**input_for_parameter_extraction.model_dump())
#         prediction5 = extract_cmd_params5(**input_for_parameter_extraction.model_dump())

#         # dspy.inspect_history(n=1)
#         print(prediction1)
#         print(prediction2)
#         print(prediction3)
#         print(prediction4)
#         print(prediction5)
