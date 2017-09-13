/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.spark.sql.execution.datasources.csv

import com.univocity.parsers.csv.CsvWriter

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.expressions.{ExpectsInputTypes, Expression, TimeZoneAwareExpression, UnaryExpression}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.util.DateTimeUtils
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

/**
 * Converts a csv input string to a [[StructType]] with the specified schema.
 *
 * TODO: Move this class into org.apache.spark.sql.catalyst.expressions in Spark-v2.2+
 */
case class CsvToStruct(
    schema: StructType,
    options: Map[String, String],
    child: Expression,
    timeZoneId: Option[String] = None)
  extends UnaryExpression with TimeZoneAwareExpression with CodegenFallback with ExpectsInputTypes {

  def this(schema: StructType, options: Map[String, String], child: Expression) =
    this(schema, options, child, None)

  override def nullable: Boolean = true

  @transient private lazy val csvOptions = new CSVOptions(options, timeZoneId.get)
  @transient private lazy val csvParser = new UnivocityParser(schema, schema, csvOptions)

  private def parse(input: String): InternalRow = csvParser.parse(input)

  override def dataType: DataType = schema

  override def nullSafeEval(csv: Any): Any = {
    try parse(csv.toString) catch { case _: RuntimeException => null }
  }

  override def inputTypes: Seq[AbstractDataType] = StringType :: Nil

  override def withTimeZone(timeZoneId: String): TimeZoneAwareExpression =
    copy(timeZoneId = Option(timeZoneId))
}

private class CsvGenerator(schema: StructType, options: CSVOptions) {

  // A `ValueConverter` is responsible for converting a value of an `InternalRow` to `String`.
  // When the value is null, this converter should not be called.
  private type ValueConverter = (InternalRow, Int) => String

  // `ValueConverter`s for all values in the fields of the schema
  private val valueConverters: Array[ValueConverter] =
    schema.map(_.dataType).map(makeConverter).toArray

  private def makeConverter(dataType: DataType): ValueConverter = dataType match {
    case DateType =>
      (row: InternalRow, ordinal: Int) =>
        options.dateFormat.format(DateTimeUtils.toJavaDate(row.getInt(ordinal)))

    case TimestampType =>
      (row: InternalRow, ordinal: Int) =>
        options.timestampFormat.format(DateTimeUtils.toJavaTimestamp(row.getLong(ordinal)))

    case udt: UserDefinedType[_] => makeConverter(udt.sqlType)

    case dt: DataType =>
      (row: InternalRow, ordinal: Int) =>
        row.get(ordinal, dt).toString
  }

  def convertRow(row: InternalRow): Seq[String] = {
    var i = 0
    val values = new Array[String](row.numFields)
    while (i < row.numFields) {
      if (!row.isNullAt(i)) {
        values(i) = valueConverters(i).apply(row, i)
      } else {
        values(i) = options.nullValue
      }
      i += 1
    }
    values
  }
}

/**
 * Converts a [[StructType]] to a csv output string.
 */
case class StructToCsv(
    options: Map[String, String],
    child: Expression,
    timeZoneId: Option[String] = None)
  extends UnaryExpression with TimeZoneAwareExpression with CodegenFallback with ExpectsInputTypes {
  override def nullable: Boolean = true

  @transient
  private lazy val params = new CSVOptions(options, timeZoneId.get)

  @transient
  private lazy val dataSchema = child.dataType.asInstanceOf[StructType]

  @transient
  private lazy val writer = new CsvGenerator(dataSchema, params)

  override def dataType: DataType = StringType

  private def verifySchema(schema: StructType): Unit = {
    def verifyType(dataType: DataType): Unit = dataType match {
      case ByteType | ShortType | IntegerType | LongType | FloatType |
           DoubleType | BooleanType | _: DecimalType | TimestampType |
           DateType | StringType =>

      case udt: UserDefinedType[_] => verifyType(udt.sqlType)

      case _ =>
        throw new UnsupportedOperationException(
          s"CSV data source does not support ${dataType.simpleString} data type.")
    }

    schema.foreach(field => verifyType(field.dataType))
  }

  override def checkInputDataTypes(): TypeCheckResult = {
    if (StructType.acceptsType(child.dataType)) {
      try {
        verifySchema(child.dataType.asInstanceOf[StructType])
        TypeCheckResult.TypeCheckSuccess
      } catch {
        case e: UnsupportedOperationException =>
          TypeCheckResult.TypeCheckFailure(e.getMessage)
      }
    } else {
      TypeCheckResult.TypeCheckFailure(
        s"$prettyName requires that the expression is a struct expression.")
    }
  }

  override def nullSafeEval(row: Any): Any = {
    val rowStr = writer.convertRow(row.asInstanceOf[InternalRow])
      .mkString(params.delimiter.toString)
    UTF8String.fromString(rowStr)
  }

  override def inputTypes: Seq[AbstractDataType] = StructType :: Nil

  override def withTimeZone(timeZoneId: String): TimeZoneAwareExpression =
    copy(timeZoneId = Option(timeZoneId))
}
