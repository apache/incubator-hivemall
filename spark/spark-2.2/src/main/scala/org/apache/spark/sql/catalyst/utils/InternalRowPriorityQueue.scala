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

package org.apache.spark.sql.catalyst.utils

import java.io.Serializable
import java.util.{PriorityQueue => JPriorityQueue}

import scala.collection.JavaConverters._
import scala.collection.generic.Growable

import org.apache.spark.sql.catalyst.InternalRow

private[sql] class InternalRowPriorityQueue(
    maxSize: Int,
    compareFunc: (Any, Any) => Int
  ) extends Iterable[(Any, InternalRow)] with Growable[(Any, InternalRow)] with Serializable {

  private[this] val ordering = new Ordering[(Any, InternalRow)] {
    override def compare(x: (Any, InternalRow), y: (Any, InternalRow)): Int =
      compareFunc(x._1, y._1)
  }

  private val underlying = new JPriorityQueue[(Any, InternalRow)](maxSize, ordering)

  override def iterator: Iterator[(Any, InternalRow)] = underlying.iterator.asScala

  override def size: Int = underlying.size

  override def ++=(xs: TraversableOnce[(Any, InternalRow)]): this.type = {
    xs.foreach { this += _ }
    this
  }

  override def +=(elem: (Any, InternalRow)): this.type = {
    if (size < maxSize) {
      underlying.offer((elem._1, elem._2.copy()))
    } else {
      maybeReplaceLowest(elem)
    }
    this
  }

  override def +=(elem1: (Any, InternalRow), elem2: (Any, InternalRow), elems: (Any, InternalRow)*)
    : this.type = {
    this += elem1 += elem2 ++= elems
  }

  override def clear() { underlying.clear() }

  private def maybeReplaceLowest(a: (Any, InternalRow)): Boolean = {
    val head = underlying.peek()
    if (head != null && ordering.gt(a, head)) {
      underlying.poll()
      underlying.offer((a._1, a._2.copy()))
    } else {
      false
    }
  }
}
